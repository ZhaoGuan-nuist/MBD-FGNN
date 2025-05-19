import os
import json
import numpy as np
import matplotlib.pyplot as plt
from data.DataReader import GraphDataReader
from data.datasplit import *
from models.models import *


def visualize_accuracy_evolution(accuracy_history, dataset_name):
    if not accuracy_history:
        print("No accuracy history to visualize.")
        return 0, 0

    rounds, accuracies = zip(*accuracy_history)
    best_round_idx = np.argmax(accuracies)
    best_round = rounds[best_round_idx]
    best_acc = accuracies[best_round_idx]

    plt.figure(figsize=(12, 8))
    plt.plot(rounds, accuracies, marker='o', linewidth=2, markersize=2, color='#3366cc')
    plt.scatter([best_round], [best_acc], color='red', s=100, zorder=5)

    plt.title(f'Model Accuracy Evolution on {dataset_name} (noise=0)', fontsize=20)
    plt.xlabel('Round', fontsize=18)
    plt.ylabel('Test Accuracy', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.annotate(f'Best: {best_acc:.4f}\nRound: {best_round}',
                 xy=(best_round, best_acc),
                 xytext=(best_round + 2, best_acc - 0.05),
                 fontsize=14,
                 arrowprops=dict(facecolor='black', shrink=0.05, width=2),
                 bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.8))

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # 确保目录存在
    os.makedirs('explanation_data', exist_ok=True)

    plt.savefig(f'explanation_data/accuracy_evolution_{dataset_name}.png', dpi=300)
    plt.close()

    print(f"Accuracy evolution plot saved to explanation_data/accuracy_evolution_{dataset_name}.png")

    return best_round, best_acc


def analyze_importance_evolution(explanation_history, dataset_name):

    if not explanation_history:
        print("No explanation history to analyze.")
        return None

    print("\n=== Dendrite Importance Evolution Analysis ===")

    all_rounds_importance = []
    for round_data in explanation_history:
        if not round_data or 'importance' not in round_data[0]:
            continue

        round_avg = []
        for i in range(len(round_data[0]['importance'])):
            branch_imp = [exp['importance'][i] for exp in round_data if 'importance' in exp]
            round_avg.append(float(np.mean(branch_imp)))
        all_rounds_importance.append(round_avg)

    os.makedirs('explanation_data', exist_ok=True)

    with open(f'explanation_data/importance_evolution_{dataset_name}.json', 'w') as f:
        json.dump(all_rounds_importance, f, indent=2)

    if all_rounds_importance:
        plt.figure(figsize=(12, 8))
        rounds = list(range(1, len(all_rounds_importance) + 1))
        for branch_idx in range(len(all_rounds_importance[0])):
            branch_importance = [round_data[branch_idx] for round_data in all_rounds_importance]
            plt.plot(rounds, branch_importance, linewidth=2, marker='o', markersize=2,
                     label=f'Branch {branch_idx}')

        plt.xlabel('Round', fontsize=16)
        plt.ylabel('Branch Importance', fontsize=16)
        plt.title(f'Dendrite Branch Importance Evolution on {dataset_name}(noise = 0)', fontsize=18)
        plt.legend(fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.savefig(f'explanation_data/importance_evolution_{dataset_name}.png', dpi=300)
        plt.close()

        print(f"Importance evolution plot saved to explanation_data/importance_evolution_{dataset_name}.png")

    return all_rounds_importance


def save_explanation_to_json(explanation, filename):

    if explanation is None or 'importance' not in explanation:
        print(f"No valid explanation data to save to {filename}")
        return

    importance_data = {f"branch_{i}_importance": float(imp) for i, imp in enumerate(explanation['importance'])}

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'w') as f:
        json.dump(importance_data, f, indent=2)

    print(f"Explanation data saved to {filename}")


def print_final_results(final_explanation, final_acc, best_acc, best_round, dataset_name):

    print(f"Final Test Accuracy on {dataset_name}: {final_acc:.4f}")
    print(f"Best Test Accuracy on {dataset_name}: {best_acc:.4f}")
    print(f"Best round on {dataset_name}: {best_round}")


    if final_explanation is not None and 'importance' in final_explanation:
        print("\nFinal Model Dendrite Importance:")
        for i, imp in enumerate(final_explanation['importance']):
            print(f"  Branch {i}: {imp:.4f}")

        save_explanation_to_json(
            final_explanation,
            f'explanation_data/final_model_explanation_{dataset_name}.json'
        )


def run_privacy_comparison(dataset_name, config, device):

    from main_v3 import evaluate,client_train_with_adaptive_dp
    import copy
    import random
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    noise_multipliers = [0.0, 0.1, 0.2, 0.4]  # 可以根据需要修改

    print(f"\n=== Running privacy comparison on dataset: {dataset_name} ===")

    data = GraphDataReader.load_data(
        name=dataset_name,
        **config['dataset_args'].get(dataset_name, {})
    )

    all_noise_accuracy_history = {}

    for noise_multiplier in noise_multipliers:
        print(f"\n--- Running with noise_multiplier = {noise_multiplier} ---")

        current_config = config.copy()
        current_config['noise_multiplier'] = noise_multiplier

        global_model = DDGAT(data.metadata).to(device)
        clients_data = split_graph_data(data, current_config['num_clients'])
        accuracy_history = []
        explanation_history = []

        for round in range(current_config['global_rounds']):
            selected_clients = random.sample(range(current_config['num_clients']),
                                             max(1, current_config['clients_per_round']))
            updates = []
            round_explanations = []

            for client_id in selected_clients:
                update = client_train_with_adaptive_dp(
                    copy.deepcopy(global_model),
                    clients_data[client_id],
                    epochs=current_config['local_epochs'],
                    lr_local=current_config['lr_local'],
                    base_noise=current_config['noise_multiplier']
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

            if (round + 1) % current_config['test_every'] == 0:
                acc, test_explanation = evaluate(global_model, data, data.test_mask)
                print(f"Round {round + 1}: Test Acc = {acc:.4f}")
                accuracy_history.append((round + 1, acc))

            if len(round_explanations) > 0:
                folder_name = f"noise_{noise_multiplier}"
                os.makedirs(f"explanation_data/{folder_name}", exist_ok=True)
                explanation_history.append(round_explanations)

        all_noise_accuracy_history[noise_multiplier] = accuracy_history

    if len(all_noise_accuracy_history) > 0:
        plt.figure(figsize=(14, 10))

        colors = ['#3366cc', '#dc3912', '#ff9900', '#109618', '#990099']
        markers = ['o', 's', '^', 'x', 'D']

        best_configs = []

        for i, (noise_level, history) in enumerate(all_noise_accuracy_history.items()):
            if len(history) == 0:
                continue

            rounds, accuracies = zip(*history)
            best_round_idx = np.argmax(accuracies)
            best_round = rounds[best_round_idx]
            best_acc = accuracies[best_round_idx]

            best_configs.append((noise_level, best_round, best_acc))

            color_idx = i % len(colors)
            marker_idx = i % len(markers)

            label = 'No Privacy Protection' if noise_level == 0.0 else f'DP (noise = {noise_level})'

            plt.plot(rounds, accuracies,
                     marker=markers[marker_idx],
                     markersize=2,
                     linewidth=2,
                     color=colors[color_idx],
                     label=label)

            plt.scatter([best_round], [best_acc],
                        s=100,
                        color=colors[color_idx],
                        edgecolor='black',
                        zorder=5)

        plt.title(f'Model Accuracy with Different Privacy Settings ({dataset_name})',
                  fontsize=20)
        plt.xlabel('Round', fontsize=18)
        plt.ylabel('Test Accuracy', fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=14, loc='lower right')

        for noise_level, best_round, best_acc in best_configs:
            label = f'Best: {best_acc:.4f}'
            plt.annotate(label,
                         xy=(best_round, best_acc),
                         xytext=(best_round + 1, best_acc - 0.03),
                         fontsize=12,
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

        plt.tight_layout()
        save_path = f'explanation_data/privacy_accuracy_comparison_{dataset_name}.png'
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"Privacy accuracy comparison plot saved to {save_path}")
        return save_path

    return None


def visualize_branch_activations(model, data, sample_nodes=10):

    model.eval()
    device = next(model.parameters()).device

    if not isinstance(data, dict):
        data = data.to(device)
        y = data.y
    else:
        for key in data:
            if torch.is_tensor(data[key]):
                data[key] = data[key].to(device)
        y = data['y']
        from torch_geometric.data import Data
        data_obj = Data(x=data['x'], edge_index=data['edge_index'])
        data = data_obj

    unique_classes = torch.unique(y)
    selected_nodes = []

    for cls in unique_classes:
        class_nodes = torch.where(y == cls)[0]
        if len(class_nodes) > 0:
            samples_per_class = max(1, sample_nodes // len(unique_classes))
            selected = class_nodes[torch.randperm(len(class_nodes))[:min(samples_per_class, len(class_nodes))]]
            selected_nodes.extend(selected.tolist())

    if not selected_nodes:
        print("No nodes were selected for activation visualization.")
        return

    with torch.no_grad():
        _ = model(data)
        dendrite_explanation = model.explain_dendrites()

    if dendrite_explanation is None or 'activations' not in dendrite_explanation:
        print("No activation information available in the explanation.")
        return

    activations = dendrite_explanation.get('activations', [])
    print(f"Activation data type: {type(activations)}")

    if isinstance(activations, list):
        print(f"Number of activation elements: {len(activations)}")
        if len(activations) > 0:
            first_elem = activations[0]
            print(f"First element type: {type(first_elem)}")
            if torch.is_tensor(first_elem):
                print(f"First element shape: {first_elem.shape}")
    else:
        print(f"Activations is not a list but: {type(activations)}")

    node_activations = []
    node_labels = []

    try:
        if isinstance(activations, dict):
            if 'node_activations' in activations:
                for node_idx in selected_nodes:
                    if node_idx < len(activations['node_activations']):
                        act = activations['node_activations'][node_idx]
                        if torch.is_tensor(act):
                            node_activations.append(act.cpu().numpy())
                        else:
                            node_activations.append(act)
                        node_labels.append(y[node_idx].item())
            else:
                print("Activation dictionary does not contain 'node_activations' key")
                return

        elif isinstance(activations, list) and len(activations) > 0 and torch.is_tensor(activations[0]):
            first_act = activations[0]
            if len(first_act.shape) == 1:
                num_branches = len(activations)
                for node_idx in selected_nodes:
                    node_act = []
                    for branch_act in activations:
                        if node_idx < branch_act.shape[0]:
                            node_act.append(branch_act[node_idx].item())
                    if node_act:
                        node_activations.append(node_act)
                        node_labels.append(y[node_idx].item())
            else:
                num_branches = activations[0].shape[1] if len(activations[0].shape) > 1 else 1
                for node_idx in selected_nodes:
                    node_act = []
                    for branch_idx, branch_act in enumerate(activations):
                        if node_idx < branch_act.shape[0]:
                            if len(branch_act.shape) > 1:
                                node_act.extend(branch_act[node_idx].cpu().numpy().flatten())
                            else:
                                node_act.append(branch_act[node_idx].item())
                    if node_act:
                        node_activations.append(node_act)
                        node_labels.append(y[node_idx].item())

        elif isinstance(activations, list) and len(activations) > 0:
            for node_idx in selected_nodes:
                if node_idx < len(activations):
                    act = activations[node_idx]
                    if torch.is_tensor(act):
                        node_activations.append(act.cpu().numpy())
                    else:
                        node_activations.append(act)
                    node_labels.append(y[node_idx].item())

        else:
            print(f"Unknown activation data structure: {type(activations)}")
            print(f"Sample activation data: {activations[:2] if isinstance(activations, list) else activations}")
            return

    except Exception as e:
        print(f"Error processing activation data: {e}")
        import traceback
        traceback.print_exc()
        return

    if not node_activations:
        print("No activation data could be collected for the selected nodes.")
        return

    print(f"Collected activations: {len(node_activations)} nodes")
    print(f"First node activation shape: {np.array(node_activations[0]).shape}")

    max_len = max(len(np.array(act).flatten()) for act in node_activations)
    uniform_activations = []
    for act in node_activations:
        act_array = np.array(act).flatten()
        if len(act_array) < max_len:
            padded = np.pad(act_array, (0, max_len - len(act_array)), 'constant')
            uniform_activations.append(padded)
        else:
            uniform_activations.append(act_array)

    activation_matrix = np.array(uniform_activations)
    print(f"Final activation matrix shape: {activation_matrix.shape}")

    if activation_matrix.size == 0 or len(activation_matrix.shape) < 2:
        print(f"Invalid activation matrix shape: {activation_matrix.shape}. Cannot create heatmap.")
        return

    max_cols = min(30, activation_matrix.shape[1])
    if activation_matrix.shape[1] > max_cols:
        activation_matrix = activation_matrix[:, :max_cols]
        print(f"Truncated activation matrix to {activation_matrix.shape} for visualization")

    plt.figure(figsize=(16, 10))
    im = plt.imshow(activation_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(im, label='Activation Strength')

    plt.xlabel('Dendrite Branch Feature Index', fontsize=14)
    plt.ylabel('Sample Node Index', fontsize=14)
    plt.title('Dendrite Activation Patterns Across Different Nodes', fontsize=16)

    plt.yticks(range(len(node_labels)),
               [f"Node {selected_nodes[i]} (Class {lbl})" for i, lbl in enumerate(node_labels)],
               fontsize=10)

    plt.tight_layout()
    save_path = f'explanation_data/branch_activations.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Branch activation visualization saved to {save_path}")
    return save_path


def analyze_branch_class_correlation(model, data):

    model.eval()
    device = next(model.parameters()).device
    if not isinstance(data, dict):
        data = data.to(device)
        y = data.y
        test_mask = data.test_mask if hasattr(data, 'test_mask') else None
    else:
        for key in data:
            if torch.is_tensor(data[key]):
                data[key] = data[key].to(device)
        y = data['y']
        test_mask = data['test_mask'] if 'test_mask' in data else None
        from torch_geometric.data import Data
        data_obj = Data(x=data['x'], edge_index=data['edge_index'])
        data = data_obj

    if test_mask is None:
        test_mask = torch.ones_like(y, dtype=torch.bool)

    test_labels = y[test_mask].cpu().numpy()
    test_node_indices = torch.where(test_mask)[0]

    test_activations = []
    with torch.no_grad():
        _ = model(data)

        dendrite_explanation = model.explain_dendrites()

        if dendrite_explanation is None or 'activations' not in dendrite_explanation:
            print("No activation information available in the explanation.")
            return

    activations = dendrite_explanation.get('activations', [])
    print(f"Activation data type: {type(activations)}")

    if isinstance(activations, list):
        print(f"Number of activation elements: {len(activations)}")
        if len(activations) > 0:
            first_elem = activations[0]
            print(f"First element type: {type(first_elem)}")
            if torch.is_tensor(first_elem):
                print(f"First element shape: {first_elem.shape}")
    else:
        print(f"Activations is not a list but: {type(activations)}")

    try:
        if isinstance(activations, dict):
            if 'node_activations' in activations:
                for node_idx in test_node_indices:
                    if node_idx < len(activations['node_activations']):
                        act = activations['node_activations'][node_idx]
                        if torch.is_tensor(act):
                            test_activations.append(act.cpu().numpy())
                        else:
                            test_activations.append(act)
            else:
                print("Activation dictionary does not contain 'node_activations' key")
                return

        elif isinstance(activations, list) and len(activations) > 0 and torch.is_tensor(activations[0]):
            first_act = activations[0]
            if len(first_act.shape) == 1:
                num_branches = len(activations)
                for node_idx in test_node_indices:
                    node_act = []
                    for branch_act in activations:
                        if node_idx < branch_act.shape[0]:
                            node_act.append(branch_act[node_idx].item())
                    if node_act:
                        test_activations.append(node_act)
            else:
                num_branches = activations[0].shape[1] if len(activations[0].shape) > 1 else 1
                for node_idx in test_node_indices:
                    node_act = []
                    for branch_idx, branch_act in enumerate(activations):
                        if node_idx < branch_act.shape[0]:
                            if len(branch_act.shape) > 1:
                                node_act.extend(branch_act[node_idx].cpu().numpy().flatten())
                            else:
                                node_act.append(branch_act[node_idx].item())
                    if node_act:
                        test_activations.append(node_act)

        elif isinstance(activations, list) and len(activations) > 0:
            for node_idx in test_node_indices:
                if node_idx < len(activations):
                    act = activations[node_idx]
                    if torch.is_tensor(act):
                        test_activations.append(act.cpu().numpy())
                    else:
                        test_activations.append(act)

        else:
            print(f"Unknown activation data structure: {type(activations)}")
            print(f"Sample activation data: {activations[:2] if isinstance(activations, list) else activations}")
            return

    except Exception as e:
        print(f"Error processing activation data: {e}")
        import traceback
        traceback.print_exc()
        return

    if not test_activations:
        print("No activation data could be collected for the test nodes.")
        return

    print(f"Collected activations: {len(test_activations)} test nodes")
    print(f"First node activation shape: {np.array(test_activations[0]).shape}")

    max_len = max(len(np.array(act).flatten()) for act in test_activations)
    uniform_activations = []
    for act in test_activations:
        act_array = np.array(act).flatten()
        if len(act_array) < max_len:
            padded = np.pad(act_array, (0, max_len - len(act_array)), 'constant')
            uniform_activations.append(padded)
        else:
            uniform_activations.append(act_array)

    test_activations = np.array(uniform_activations)
    print(f"Final test activations shape: {test_activations.shape}")

    num_branches = test_activations.shape[1]
    print(f"Number of branches/features available: {num_branches}")
    unique_classes = np.unique(test_labels)

    class_avg_activations = {}
    for cls in unique_classes:
        cls_indices = np.where(test_labels == cls)[0]
        if len(cls_indices) > 0:
            class_avg_activations[cls] = np.mean(test_activations[cls_indices], axis=0)

    class_top_branches = {}
    for cls, activations_array in class_avg_activations.items():
        top_count = min(3, len(activations_array))

        top_indices = np.argsort(activations_array)[-top_count:][::-1]

        valid_indices = [idx for idx in top_indices if idx < len(activations_array)]

        class_top_branches[cls] = [(branch, activations_array[branch]) for branch in valid_indices]

    plt.figure(figsize=(16, 10))

    correlation_matrix = np.zeros((len(unique_classes), num_branches))

    for i, cls in enumerate(unique_classes):
        if cls in class_avg_activations:
            correlation_matrix[i] = class_avg_activations[cls]

    im = plt.imshow(correlation_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(im, label='Average Activation')

    plt.xlabel('Dendrite Feature Index', fontsize=14)
    plt.yticks(range(len(unique_classes)), [f'Class {cls}' for cls in unique_classes], fontsize=12)
    plt.title('Average Dendrite Feature Activation per Class', fontsize=16)

    for i, cls in enumerate(unique_classes):
        if cls in class_top_branches:
            for j, (branch, value) in enumerate(class_top_branches[cls]):
                if 0 <= branch < correlation_matrix.shape[1] and 0 <= i < correlation_matrix.shape[0]:
                    plt.text(branch, i, f"{value:.2f}",
                             ha='center', va='center',
                             color='white' if value < 0.7 * np.max(correlation_matrix) else 'black',
                             fontweight='bold', fontsize=10)

    plt.tight_layout()
    save_path = f'explanation_data/branch_class_correlation.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print("\nClass-specific feature importance:")
    for cls, branches in class_top_branches.items():
        print(
            f"Class {cls}: Top features {[b[0] for b in branches]} (activations: {[f'{b[1]:.2f}' for b in branches]})")

    print(f"Branch-class correlation analysis saved to {save_path}")
    return save_path


def visualize_node_neighborhood_activations(model, data, sample_nodes=5, neighborhood_size=10):

    model.eval()
    device = next(model.parameters()).device

    if not isinstance(data, dict):
        data = data.to(device)
        y = data.y
        edge_index = data.edge_index
    else:
        for key in data:
            if torch.is_tensor(data[key]):
                data[key] = data[key].to(device)
        y = data['y']
        edge_index = data['edge_index']
        from torch_geometric.data import Data
        data_obj = Data(x=data['x'], edge_index=data['edge_index'])
        data = data_obj

    unique_classes = torch.unique(y)
    selected_nodes = []

    for cls in unique_classes:
        class_nodes = torch.where(y == cls)[0]
        if len(class_nodes) > 0:
            samples_per_class = max(1, min(2, sample_nodes // len(unique_classes)))
            selected = class_nodes[torch.randperm(len(class_nodes))[:samples_per_class]]
            selected_nodes.extend(selected.tolist())

    if not selected_nodes:
        print("No nodes were selected for neighborhood visualization.")
        return

    with torch.no_grad():
        _ = model(data)

        dendrite_explanation = model.explain_dendrites()

    if dendrite_explanation is None or 'activations' not in dendrite_explanation:
        print("No activation information available in the explanation.")
        return

    activations = dendrite_explanation.get('activations', [])
    print(f"Activation data type: {type(activations)}")

    if isinstance(activations, list):
        print(f"Number of activation elements: {len(activations)}")
        if len(activations) > 0:
            first_elem = activations[0]
            print(f"First element type: {type(first_elem)}")
            if torch.is_tensor(first_elem):
                print(f"First element shape: {first_elem.shape}")

    fig, axes = plt.subplots(len(selected_nodes), 1, figsize=(16, 5 * len(selected_nodes)))
    if len(selected_nodes) == 1:
        axes = [axes]

    for i, center_node in enumerate(selected_nodes):
        src, dst = edge_index.cpu().numpy()

        neighbors_idx = np.where(src == center_node)[0]
        if len(neighbors_idx) == 0:
            neighbors_idx = np.where(dst == center_node)[0]
            neighbors = src[neighbors_idx]
        else:
            neighbors = dst[neighbors_idx]

        if len(neighbors) > neighborhood_size:
            neighbors = neighbors[:neighborhood_size]


        neighbors = neighbors.tolist()
        if center_node not in neighbors:
            neighbors = [center_node] + neighbors

        node_labels = []
        for node in neighbors:
            if 0 <= node < len(y):
                node_labels.append(y[node].item())
            else:
                node_labels.append(-1)

        neighborhood_activations = []

        try:
            if isinstance(activations, dict):
                if 'node_activations' in activations:
                    for node in neighbors:
                        if 0 <= node < len(activations['node_activations']):
                            act = activations['node_activations'][node]
                            if torch.is_tensor(act):
                                neighborhood_activations.append(act.cpu().numpy())
                            else:
                                neighborhood_activations.append(act)
                else:
                    print("Activation dictionary does not contain 'node_activations' key")
                    continue

            elif isinstance(activations, list) and len(activations) > 0 and torch.is_tensor(activations[0]):
                first_act = activations[0]
                if len(first_act.shape) == 1:
                    num_branches = len(activations)

                    for node in neighbors:
                        node_act = []
                        for branch_act in activations:
                            if 0 <= node < branch_act.shape[0]:
                                node_act.append(branch_act[node].item())
                        if node_act:
                            neighborhood_activations.append(node_act)
                else:
                    for node in neighbors:
                        node_act = []
                        for branch_idx, branch_act in enumerate(activations):
                            if 0 <= node < branch_act.shape[0]:
                                if len(branch_act.shape) > 1:
                                    node_act.extend(branch_act[node].cpu().numpy().flatten())
                                else:
                                    node_act.append(branch_act[node].item())
                        if node_act:
                            neighborhood_activations.append(node_act)

            elif isinstance(activations, list) and len(activations) > 0:
                for node in neighbors:
                    if 0 <= node < len(activations):
                        act = activations[node]
                        if torch.is_tensor(act):
                            neighborhood_activations.append(act.cpu().numpy())
                        else:
                            neighborhood_activations.append(act)

            else:
                print(f"Unknown activation data structure for node {center_node}")
                continue

        except Exception as e:
            print(f"Error processing activation data for node {center_node}: {e}")
            import traceback
            traceback.print_exc()
            continue

        if not neighborhood_activations:
            print(f"No activation data could be collected for node {center_node} and its neighbors.")
            continue

        max_len = max(len(np.array(act).flatten()) for act in neighborhood_activations)
        uniform_activations = []
        for act in neighborhood_activations:
            act_array = np.array(act).flatten()
            if len(act_array) < max_len:
                padded = np.pad(act_array, (0, max_len - len(act_array)), 'constant')
                uniform_activations.append(padded)
            else:
                uniform_activations.append(act_array)

        activation_matrix = np.array(uniform_activations)

        max_cols = min(30, activation_matrix.shape[1])
        if activation_matrix.shape[1] > max_cols:
            activation_matrix = activation_matrix[:, :max_cols]

        ax = axes[i]
        im = ax.imshow(activation_matrix, cmap='viridis', aspect='auto')


        fig.colorbar(im, ax=ax, label='Activation Strength')

        ax.set_xlabel('Dendrite Feature Index', fontsize=12)

        node_labels_text = [f"Node {node} (Class {label})" for node, label in zip(neighbors, node_labels)]
        ax.set_yticks(range(len(node_labels_text)))
        ax.set_yticklabels(node_labels_text, fontsize=10)

        ax.set_title(f'Dendrite Activation Patterns for Node {center_node} and its Neighbors', fontsize=14)

    plt.tight_layout()
    save_path = f'explanation_data/neighborhood_activations.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Neighborhood activation visualization saved to {save_path}")
    return save_path

def visualize_3d_branch_activations(model, data):
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    model.eval()

    features = data.x.cpu().numpy()
    labels = data.y.cpu().numpy()

    pca = PCA(n_components=3)
    node_3d = pca.fit_transform(features)

    with torch.no_grad():
        _ = model(data)
        dendrite_explanation = model.explain_dendrites()

    if dendrite_explanation is None or 'activations' not in dendrite_explanation or 'importance' not in dendrite_explanation:
        print("Required explanation data not available.")
        return

    branch_importance = dendrite_explanation['importance']
    top_branch_idx = np.argmax(branch_importance)

    branch_activations = [act[top_branch_idx] if i < len(dendrite_explanation['activations']) else 0
                          for i, act in enumerate(dendrite_explanation['activations'])]

    if max(branch_activations) > min(branch_activations):
        normalized_acts = [(act - min(branch_activations)) / (max(branch_activations) - min(branch_activations))
                           for act in branch_activations]
    else:
        normalized_acts = [0.5] * len(branch_activations)

    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    unique_labels = np.unique(labels)
    markers = ['o', 's', '^', 'P', '*']

    for i, label in enumerate(unique_labels):
        idx = labels == label
        scatter = ax.scatter(
            node_3d[idx, 0],
            node_3d[idx, 1],
            node_3d[idx, 2],
            c=[normalized_acts[j] for j in np.where(idx)[0]],
            cmap='viridis',
            marker=markers[i % len(markers)],
            s=80,
            alpha=0.7,
            label=f'Class {label}'
        )

    fig.colorbar(scatter, ax=ax, label=f'Branch {top_branch_idx} Activation')

    ax.set_title(
        f'3D Visualization of Branch {top_branch_idx} Activation (Importance: {branch_importance[top_branch_idx]:.4f})',
        fontsize=16)
    ax.set_xlabel('PCA Component 1', fontsize=14)
    ax.set_ylabel('PCA Component 2', fontsize=14)
    ax.set_zlabel('PCA Component 3', fontsize=14)
    ax.legend(fontsize=12)

    plt.savefig(f'explanation_data/3d_branch_{top_branch_idx}_activation.png', dpi=300)
    plt.close()

    print(f"3D visualization for top branch saved to explanation_data/")




def analyze_explanation_stability_vs_performance(explanation_history, accuracy_history):

    if not explanation_history or not accuracy_history:
        print("Insufficient data for stability vs performance analysis.")
        return

    explanation_changes = []

    for i in range(1, len(explanation_history)):
        prev_exp = explanation_history[i - 1]
        curr_exp = explanation_history[i]

        changes = []
        for prev_client_exp, curr_client_exp in zip(prev_exp, curr_exp):
            if 'importance' in prev_client_exp and 'importance' in curr_client_exp:
                prev_imp = np.array(prev_client_exp['importance'])
                curr_imp = np.array(curr_client_exp['importance'])

                if np.all(prev_imp == 0) or np.all(curr_imp == 0):
                    changes.append(0)
                else:
                    similarity = np.dot(prev_imp, curr_imp) / (np.linalg.norm(prev_imp) * np.linalg.norm(curr_imp))
                    distance = 1 - similarity
                    changes.append(distance)

        if changes:
            explanation_changes.append(np.mean(changes))
        else:
            explanation_changes.append(0)

    rounds, accuracies = zip(*accuracy_history)

    min_length = min(len(explanation_changes), len(accuracies) - 1)
    explanation_changes = explanation_changes[:min_length]
    accuracies_changes = [accuracies[i + 1] - accuracies[i] for i in range(min_length)]

    fig, ax1 = plt.subplots(figsize=(12, 8))

    color = 'tab:blue'
    ax1.set_xlabel('Training Round', fontsize=14)
    ax1.set_ylabel('Explanation Change (Cosine Distance)', color=color, fontsize=14)
    ax1.plot(range(1, min_length + 1), explanation_changes, color=color, marker='o', markersize=5)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Accuracy Change', color=color, fontsize=14)
    ax2.plot(range(1, min_length + 1), accuracies_changes, color=color, marker='s', markersize=5)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Explanation Stability vs. Performance Changes', fontsize=16)
    ax1.grid(True, linestyle='--', alpha=0.7)

    fig.tight_layout()
    plt.savefig('explanation_data/explanation_stability_vs_performance.png', dpi=300)
    plt.close()

    plt.figure(figsize=(10, 8))
    plt.scatter(explanation_changes, accuracies_changes, alpha=0.7, s=80)

    if len(explanation_changes) > 1:
        z = np.polyfit(explanation_changes, accuracies_changes, 1)
        p = np.poly1d(z)
        plt.plot(explanation_changes, p(explanation_changes), "r--", alpha=0.8,
                 linewidth=2, label=f'Trend: y={z[0]:.4f}x+{z[1]:.4f}')

    plt.xlabel('Explanation Change', fontsize=14)
    plt.ylabel('Accuracy Change', fontsize=14)
    plt.title('Relationship Between Explanation Stability and Performance', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('explanation_data/explanation_vs_performance_scatter.png', dpi=300)
    plt.close()

    print("Explanation stability vs. performance analysis saved to explanation_data/")





def visualize_improved_noise_distribution(importance_values, base_noise=0.1, num_dendrites=4):

    import matplotlib.pyplot as plt
    import numpy as np

    avg_importance = 1.0 / num_dendrites

    importance = np.linspace(0, 0.6, 100)

    relative_importance = importance / avg_importance

    noise_factors = 2.0 / (1 + np.exp(relative_importance - 1))

    noise_levels = base_noise * noise_factors

    plt.figure(figsize=(10, 6))

    plt.plot(importance, noise_levels, linewidth=2.5)

    reference_importances = [0.1, 0.2, avg_importance, 0.35, 0.5]
    reference_relative_importances = [i / avg_importance for i in reference_importances]
    reference_noise_factors = [2.0 / (1 + np.exp(r - 1)) for r in reference_relative_importances]
    reference_noise_levels = [base_noise * f for f in reference_noise_factors]

    plt.scatter(reference_importances, reference_noise_levels, color='red', s=100,
                label='Reference points')


    if importance_values:
        actual_relative_importances = [i / avg_importance for i in importance_values]
        actual_noise_factors = [2.0 / (1 + np.exp(r - 1)) for r in actual_relative_importances]
        actual_noise_levels = [base_noise * f for f in actual_noise_factors]

        plt.scatter(importance_values, actual_noise_levels, color='green', s=80, alpha=0.7,
                    label='Actual dendrite importances')


    plt.xlabel('Dendrite importance value')
    plt.ylabel('Applied noise level')
    plt.title('Adaptive differential privacy noise based on relative dendrite importance')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')


    plt.savefig('explanation_data/improved_noise_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Visualization saved as 'improved_noise_distribution.png'")

    print("\nReference values for dendrite importance vs. noise level:")
    print("---------------------------------------------------------")
    print("| Importance | Relative Importance | Noise Factor | Noise Level |")
    print("|------------|---------------------|--------------|-------------|")
    sample_importances = [0.05, 0.1, 0.15, 0.2, avg_importance, 0.3, 0.35, 0.4, 0.5]
    for imp in sample_importances:
        rel_imp = imp / avg_importance
        noise_factor = 2.0 / (1 + np.exp(rel_imp - 1))
        noise_level = base_noise * noise_factor
        print(
            f"| {imp:.2f}       | {rel_imp:.2f}                | {noise_factor:.2f}         | {noise_level:.4f}     |")



