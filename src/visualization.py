import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D
import seaborn as sns

def plot_model_graph(model, save_path=None):
    G = nx.DiGraph()

    layer_nodes = {}
    for i, size in enumerate(model.layer_sizes):
        layer_nodes[i] = []
        for j in range(size):
            node_id = f"L{i}N{j}"
            G.add_node(node_id, layer=i, pos=(i, j - size/2 + 0.5))
            layer_nodes[i].append(node_id)

    for i in range(len(model.layers)):
        weights, biases = model.layers[i].get_weights()
        for j in range(len(layer_nodes[i])):
            src = layer_nodes[i][j]
            for k in range(len(layer_nodes[i+1])):
                dest = layer_nodes[i+1][k]
                weight = weights[j, k]
                G.add_edge(src, dest, weight=weight)

    pos = nx.get_node_attributes(G, 'pos')

    plt.figure(figsize=(12, 8))

    for i in range(len(model.layer_sizes)):
        nx.draw_networkx_nodes(G, pos,
                              nodelist=layer_nodes[i],
                              node_color=f'C{i}',
                              node_size=500,
                              alpha=0.8)

    all_weights = [abs(G[a][b]['weight']) for a, b in G.edges()]
    max_weight = max(all_weights) if all_weights else 1

    for (u, v, d) in G.edges(data=True):
        weight = abs(d['weight'])
        width = 1 + 3 * weight / max(1e-9, max_weight)
        color = 'red' if d['weight'] < 0 else 'green'
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=width, alpha=0.6, edge_color=color)

    nx.draw_networkx_labels(G, pos, font_size=10)

    legend_elements = [
        Line2D([0], [0], color='green', lw=2, label='Positive Weight'),
        Line2D([0], [0], color='red', lw=2, label='Negative Weight')
    ]
    plt.legend(handles=legend_elements)

    for i in range(len(model.layer_sizes)):
        if i < len(model.layer_sizes) - 1:
            activation_info = str(model.activations[i])
            plt.text(i, model.layer_sizes[i]/2 + 1,
                     f"Layer {i}\n{model.layer_sizes[i]} neurons\nActivation: {activation_info}",
                     horizontalalignment='center', size=10, color=f'C{i}', weight='bold')
        else:
            plt.text(i, model.layer_sizes[i]/2 + 1,
                     f"Layer {i}\n{model.layer_sizes[i]} neurons",
                     horizontalalignment='center', size=10, color=f'C{i}', weight='bold')

    plt.title("Neural Network Structure")
    plt.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()

def plot_simplified_model_graph(model, save_path=None, max_nodes_per_layer=10):
    G = nx.DiGraph()

    layer_nodes = {}
    for i, size in enumerate(model.layer_sizes):
        layer_nodes[i] = []

        if size > max_nodes_per_layer:
            # Show first 3 nodes
            for j in range(3):
                node_id = f"L{i}N{j}"
                G.add_node(node_id, layer=i, pos=(i, j - 1.5))
                layer_nodes[i].append(node_id)

            #  ellipsis node
            node_id = f"L{i}..."
            G.add_node(node_id, layer=i, pos=(i, 1.5))

            #  last 3 nodes
            for j in range(3):
                node_id = f"L{i}N{size-3+j}"
                G.add_node(node_id, layer=i, pos=(i, j + 2.5))
                layer_nodes[i].append(node_id)
        else:
            #  all nodes if fewer
            for j in range(size):
                node_id = f"L{i}N{j}"
                G.add_node(node_id, layer=i, pos=(i, j - size/2 + 0.5))
                layer_nodes[i].append(node_id)

    #  representative edges
    for i in range(len(model.layers)):
        weights, biases = model.layers[i].get_weights()

        if len(layer_nodes[i]) < len(layer_nodes[i+1]):
            for src_idx, src in enumerate(layer_nodes[i]):
                target_indices = np.linspace(0, len(layer_nodes[i+1])-1, min(3, len(layer_nodes[i+1]))).astype(int)
                for tgt_idx in target_indices:
                    if tgt_idx < len(layer_nodes[i+1]):
                        dest = layer_nodes[i+1][tgt_idx]
                        weight_val = weights[src_idx % weights.shape[0], tgt_idx % weights.shape[1]] if src_idx < weights.shape[0] and tgt_idx < weights.shape[1] else 0
                        G.add_edge(src, dest, weight=weight_val)
        else:
            for tgt_idx, dest in enumerate(layer_nodes[i+1]):
                source_indices = np.linspace(0, len(layer_nodes[i])-1, min(3, len(layer_nodes[i]))).astype(int)
                for src_idx in source_indices:
                    if src_idx < len(layer_nodes[i]):
                        src = layer_nodes[i][src_idx]
                        weight_val = weights[src_idx % weights.shape[0], tgt_idx % weights.shape[1]] if src_idx < weights.shape[0] and tgt_idx < weights.shape[1] else 0
                        G.add_edge(src, dest, weight=weight_val)

    pos = nx.get_node_attributes(G, 'pos')

    plt.figure(figsize=(12, 8))

    # Draw nodes
    for i in range(len(model.layer_sizes)):
        layer_node_list = [n for n in G.nodes() if n.startswith(f"L{i}")]
        nx.draw_networkx_nodes(G, pos,
                              nodelist=layer_node_list,
                              node_color=f'C{i}',
                              node_size=500,
                              alpha=0.8)

    # Draw edges
    all_weights = [abs(G[a][b]['weight']) for a, b in G.edges()]
    max_weight = max(all_weights) if all_weights else 1

    for (u, v, d) in G.edges(data=True):
        weight = abs(d['weight'])
        width = 1 + 3 * weight / max(1e-9, max_weight)
        color = 'red' if d['weight'] < 0 else 'green'
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=width, alpha=0.6, edge_color=color)

    # Add node labels
    nx.draw_networkx_labels(G, pos, font_size=10)

    # Add legend
    legend_elements = [
        Line2D([0], [0], color='green', lw=2, label='Positive Weight'),
        Line2D([0], [0], color='red', lw=2, label='Negative Weight')
    ]
    plt.legend(handles=legend_elements)

    # Add layer information
    for i in range(len(model.layer_sizes)):
        if i < len(model.layer_sizes) - 1:
            activation_info = str(model.activations[i])
            plt.text(i, -model.layer_sizes[i]/2 - 1,
                     f"Layer {i}\n{model.layer_sizes[i]} neurons\nActivation: {activation_info}",
                     horizontalalignment='center', size=10, color=f'C{i}', weight='bold')
        else:
            plt.text(i, -model.layer_sizes[i]/2 - 1,
                     f"Layer {i}\n{model.layer_sizes[i]} neurons",
                     horizontalalignment='center', size=10, color=f'C{i}', weight='bold')

    plt.title("Simplified Neural Network Structure")
    plt.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()

def plot_weight_distributions(model, layers_to_plot=None, save_path=None):
    if layers_to_plot is None:
        layers_to_plot = list(range(len(model.layers)))

    n_layers = len(layers_to_plot)

    n_cols = min(3, n_layers)
    n_rows = (n_layers + n_cols - 1) // n_cols

    plt.figure(figsize=(15, 5 * n_rows))

    for i, layer_idx in enumerate(layers_to_plot):
        if layer_idx >= len(model.layers):
            continue

        weights, biases = model.layers[layer_idx].get_weights()

        flat_weights = weights.flatten()

        plt.subplot(n_rows, n_cols, i + 1)

        sns.histplot(flat_weights, kde=True)

        plt.title(f"Layer {layer_idx} Weight Distribution")
        plt.xlabel("Weight Value")
        plt.ylabel("Frequency")

        stats_text = f"Mean: {np.mean(flat_weights):.4f}\n" \
                    f"Std: {np.std(flat_weights):.4f}\n" \
                    f"Min: {np.min(flat_weights):.4f}\n" \
                    f"Max: {np.max(flat_weights):.4f}"

        plt.annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction',
                    verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()

def plot_gradient_distributions(model, layers_to_plot=None, save_path=None):
    if layers_to_plot is None:
        layers_to_plot = list(range(len(model.layers)))

    n_layers = len(layers_to_plot)

    n_cols = min(3, n_layers)
    n_rows = (n_layers + n_cols - 1) // n_cols

    plt.figure(figsize=(15, 5 * n_rows))

    for i, layer_idx in enumerate(layers_to_plot):
        if layer_idx >= len(model.layers):
            continue

        weight_grads, bias_grads = model.layers[layer_idx].get_gradients()

        flat_grads = weight_grads.flatten()

        plt.subplot(n_rows, n_cols, i + 1)

        sns.histplot(flat_grads, kde=True)

        plt.title(f"Layer {layer_idx} Gradient Distribution")
        plt.xlabel("Gradient Value")
        plt.ylabel("Frequency")

        stats_text = f"Mean: {np.mean(flat_grads):.4f}\n" \
                    f"Std: {np.std(flat_grads):.4f}\n" \
                    f"Min: {np.min(flat_grads):.4f}\n" \
                    f"Max: {np.max(flat_grads):.4f}"

        plt.annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction',
                    verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()

def plot_learning_curves(history, save_path=None):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(history['loss']) + 1)

    plt.plot(epochs, history['loss'], 'b', label='Training loss')
    if 'val_loss' in history and len(history['val_loss']) > 0:
        plt.plot(epochs, history['val_loss'], 'r', label='Validation loss')

    plt.title('Learning Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.grid(True, linestyle='--', alpha=0.7)

    if save_path:
        plt.savefig(save_path)

    plt.show()

def plot_prediction_comparison(y_true, y_pred, y_sklearn=None, save_path=None):
    is_classification = False
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        is_classification = True
        y_true_indices = np.argmax(y_true, axis=1)
    elif np.all(np.floor(y_true) == y_true) and len(np.unique(y_true)) < 10:
        is_classification = True
        y_true_indices = y_true.astype(int).flatten()
    else:  # Regression
        y_true_indices = y_true

    if is_classification:
        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            y_pred_indices = np.argmax(y_pred, axis=1)
        else:
            y_pred_indices = (y_pred > 0.5).astype(int).flatten()

        accuracy = np.mean(y_true_indices == y_pred_indices)

        classes = np.unique(y_true_indices)
        cm = np.zeros((len(classes), len(classes)), dtype=int)

        for i in range(len(y_true_indices)):
            cm[y_true_indices[i], y_pred_indices[i]] += 1

        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix (Accuracy: {accuracy:.4f})')
        plt.colorbar()

        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')

        thresh = cm.max() / 2
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        if y_sklearn is not None:
            if len(y_sklearn.shape) > 1 and y_sklearn.shape[1] > 1:
                y_sklearn_indices = np.argmax(y_sklearn, axis=1)
            else:
                y_sklearn_indices = (y_sklearn > 0.5).astype(int).flatten()

            sklearn_accuracy = np.mean(y_true_indices == y_sklearn_indices)
            plt.annotate(f'Sklearn Accuracy: {sklearn_accuracy:.4f}',
                         xy=(0.05, 0.05), xycoords='figure fraction',
                         horizontalalignment='left', verticalalignment='bottom',
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    else:
        # Regression case
        plt.figure(figsize=(10, 8))
        plt.scatter(y_true, y_pred, alpha=0.5, label='Custom Model')

        if y_sklearn is not None:
            plt.scatter(y_true, y_sklearn, alpha=0.5, label='Sklearn Model')

        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect Prediction')

        plt.title('Prediction Comparison')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.legend()

        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        plt.annotate(f'RMSE: {rmse:.4f}',
                    xy=(0.05, 0.95), xycoords='figure fraction',
                    horizontalalignment='left', verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

        if y_sklearn is not None:
            sklearn_rmse = np.sqrt(np.mean((y_true - y_sklearn) ** 2))
            plt.annotate(f'Sklearn RMSE: {sklearn_rmse:.4f}',
                        xy=(0.05, 0.9), xycoords='figure fraction',
                        horizontalalignment='left', verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    plt.grid(True, linestyle='--', alpha=0.7)

    if save_path:
        plt.savefig(save_path)

    plt.show()
