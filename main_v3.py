import copy
import random
from utlis.visualizations import *
from utlis.Logging import ExperimentLoggerCSV
from torch.optim import Adam


def client_train_with_dp(model, data_dict, epochs, lr_local=0, noise_multiplier=0.1, max_grad_norm=1.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    data = Data(
        x=data_dict['x'].to(device),
        edge_index=data_dict['edge_index'].to(device),
        y=data_dict['y'].to(device),
        train_mask=data_dict['train_mask'].to(device),
        val_mask=data_dict['val_mask'].to(device) if 'val_mask' in data_dict else None,
        test_mask=data_dict['test_mask'].to(device) if 'test_mask' in data_dict else None
    )

    optimizer = Adam(model.parameters(), lr=lr_local)

    importance_history = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        with torch.no_grad():
            for param in model.parameters():
                if param.grad is not None:
                    noise = torch.randn_like(param.grad) * noise_multiplier * max_grad_norm / (
                            len(data.train_mask) ** 0.5)
                    param.grad += noise

        optimizer.step()

        if epoch % 5 == 0:
            explanation = model.explain_dendrites()
            if explanation is not None:
                importance_history.append(explanation)

    dendritic_explanation = model.explain_dendrites()

    return {
        "state_dict": model.state_dict(),
        "explanation": dendritic_explanation,
        "importance_history": importance_history if len(importance_history) > 0 else None
    }


def client_train_with_adaptive_dp(model, data_dict, epochs, lr_local=0.01, base_noise=0.1, max_grad_norm=1.0,num_dendrites=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    data = Data(
        x=data_dict['x'].to(device),
        edge_index=data_dict['edge_index'].to(device),
        y=data_dict['y'].to(device),
        train_mask=data_dict['train_mask'].to(device),
        val_mask=data_dict['val_mask'].to(device) if 'val_mask' in data_dict else None,
        test_mask=data_dict['test_mask'].to(device) if 'test_mask' in data_dict else None
    )

    optimizer = Adam(model.parameters(), lr=lr_local)

    importance_history = []

    avg_importance = 1.0 / num_dendrites

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()

        explanation = model.explain_dendrites()
        branch_importance = explanation['importance'] if explanation and 'importance' in explanation else None

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is None:
                    continue

                noise_multiplier = base_noise

                if branch_importance and 'dendritic_layer.dmcu_dae.branches' in name:
                    try:
                        parts = name.split('.')
                        branch_idx = -1
                        for i, part in enumerate(parts):
                            if part == 'branches' and i + 1 < len(parts):
                                branch_idx = int(parts[i + 1])
                                break

                        if branch_idx >= 0 and branch_idx < len(branch_importance):
                            importance = branch_importance[branch_idx]

                            relative_importance = importance / avg_importance

                            noise_factor = 2.0 / (1 + np.exp(relative_importance - 1))

                            noise_multiplier = base_noise * noise_factor
                    except Exception as e:
                        print(f"Error parsing branch index: {e}, using default noise")

                noise = torch.randn_like(param.grad) * noise_multiplier * max_grad_norm / (
                        len(data.train_mask) ** 0.5)
                param.grad += noise

        optimizer.step()

        if epoch % 5 == 0 and explanation is not None:
            importance_history.append(explanation)

    dendritic_explanation = model.explain_dendrites()

    return {
        "state_dict": model.state_dict(),
        "explanation": dendritic_explanation,
        "importance_history": importance_history if len(importance_history) > 0 else None
    }


def evaluate(model, data, mask):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        logits = model(data.to(device))
        pred = logits[mask].argmax(dim=1)
        acc = (pred == data.y[mask]).sum().item() / mask.sum().item()

        explanation = model.explain_dendrites()

    return acc, explanation



def analyze_explanations(round_explanations, round_num, dataset_name):
    avg_importance = []
    for i in range(len(round_explanations[0]['importance'])):
        branch_importance = [exp['importance'][i] for exp in round_explanations]
        avg_importance.append(np.mean(branch_importance))

    os.makedirs('explanation_data', exist_ok=True)
    with open(f'explanation_data/dendrite_importance_round_{round_num}.json', 'w') as f:
        importance_data = {f"branch_{i}_importance": float(imp) for i, imp in enumerate(avg_importance)}
        json.dump(importance_data, f, indent=2)


def main():
    os.makedirs('explanation_data', exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = ExperimentLoggerCSV(log_dir="./experiment_logs", log_file="training_log_explainable.csv")

    config = {
        'datasets': ['cora'],
        'dataset_args': {
            'citeseer': {},
            'cora': {},
            'pubmed': {},
        },
        'num_clients': 5,
        'global_rounds': 100,
        'local_epochs': 5,
        'lr': 0.01,
        'lr_local': 0.01,
        'test_every': 1,
        'test_num': 5,
        'clients_per_round': 5
    }



    for dataset_name in config['datasets']:
        print(f"\n=== Training on {dataset_name} dataset ===")

        data = GraphDataReader.load_data(
            name=dataset_name,
            **config['dataset_args'].get(dataset_name, {})
        )

        clients_data = split_graph_data(data, config['num_clients'])

        client_model = DDGAT(data.metadata).to(device)
        global_model = copy.deepcopy(client_model)

        explanation_history = []
        accuracy_history = []
        best_acc = 0
        best_round = 0

        for round in range(config['global_rounds']):
            selected_clients = random.sample(range(config['num_clients']), max(1, config['test_num']))
            print(f"Round {round + 1}: Selected clients: {selected_clients}")

            updates = []
            round_explanations = []

            for client_id in selected_clients:
                update = client_train_with_adaptive_dp(
                    copy.deepcopy(global_model),
                    clients_data[client_id],
                    epochs=config['local_epochs'],
                    lr_local=config['lr_local']
                )
                updates.append(update)

                if 'explanation' in update and update['explanation'] is not None:
                    round_explanations.append(update['explanation'])

            with torch.no_grad():
                global_state_dict = global_model.state_dict()
                for key in global_state_dict:
                    global_state_dict[key] = torch.stack(
                        [update['state_dict'][key] for update in updates], 0
                    ).mean(0)
                global_model.load_state_dict(global_state_dict)

            if len(round_explanations) > 0:
                analyze_explanations(round_explanations, round + 1, dataset_name)
                explanation_history.append(round_explanations)
                all_importance_values = []
                for exp in round_explanations:
                    if 'importance' in exp:
                        all_importance_values.extend(exp['importance'])


                if  round == config['global_rounds'] - 1:
                    visualize_improved_noise_distribution(
                        all_importance_values,
                        base_noise=0.1,
                        num_dendrites=4
                    )

            if (round + 1) % config['test_every'] == 0:
                acc, test_explanation = evaluate(global_model, data, data.test_mask)
                print(f"Round {round + 1}: Test Acc = {acc:.4f}")
                accuracy_history.append((round + 1, acc))

                if acc > best_acc:
                    best_acc = acc
                    best_round = round + 1

                metrics = {
                    "round": round + 1,
                    "test_accuracy": acc
                }

                additional_info = {
                    "dataset": dataset_name,
                    "model": global_model.__class__.__name__,
                    "learning_rate": config['lr'],
                    "local_learning_rate": config['lr_local']
                }
                logger.log(experiment_name="Explainable Federated Training", metrics=metrics,
                           additional_info=additional_info)

                # 保存测试解释数据
                if test_explanation is not None and 'importance' in test_explanation:
                    save_explanation_to_json(
                        test_explanation,
                        f'explanation_data/test_explanation_round_{round + 1}.json'
                    )


        best_round, best_acc = visualize_accuracy_evolution(accuracy_history, dataset_name)
        analyze_importance_evolution(explanation_history, dataset_name)
        final_acc, final_explanation = evaluate(global_model, data, data.test_mask)
        print_final_results(final_explanation, final_acc, best_acc, best_round, dataset_name)
        #run_privacy_comparison(dataset_name, config, device)


        if explanation_history:
            print("\n=== Generating Additional Visualizations ===")
            visualize_branch_activations(global_model, data)
            analyze_branch_class_correlation(global_model, data)
            visualize_node_neighborhood_activations(global_model, data)
            if accuracy_history:
                analyze_explanation_stability_vs_performance(explanation_history, accuracy_history)
            visualize_3d_branch_activations(global_model, data)


        metrics = {
            "round": config['global_rounds'],
            "test_accuracy": final_acc
        }

        additional_info = {
            "dataset": dataset_name,
            "model": global_model.__class__.__name__,
            "learning_rate": config['lr']
        }
        logger.log(experiment_name="Explainable Federated Training - Final", metrics=metrics,
                   additional_info=additional_info)





if __name__ == "__main__":
    main()