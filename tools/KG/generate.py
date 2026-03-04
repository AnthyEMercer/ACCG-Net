import sys, os
sys.path.append(os.getcwd())

import torch
from torch.optim import AdamW
from torch_geometric.loader import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
import os
import random
import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

import config
from tools.KG.KG_Dataset import XRayGraphDataset
from model.Generator import ReportGenerator
TRAIN_NODE_FILE = "./data/node_feat_train"
VAL_NODE_FILE = "./data/node_feat_val"
TEST_NODE_FILE = "./data/node_feat_test"

INVERSE_ANA = dict((v, k) for k, v in config.ANATOMICAL_REGIONS.items())

DEVICE = torch.device("cuda:1")

def generate():
    tokenizer = T5Tokenizer.from_pretrained("")
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token

    # train_dataset = XRayGraphDataset(TRAIN_NODE_FILE, tokenizer)
    val_dataset = XRayGraphDataset(VAL_NODE_FILE, tokenizer)
    test_dataset = XRayGraphDataset(TEST_NODE_FILE, tokenizer)
    
    # train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    model = ReportGenerator(tokenizer).to(DEVICE)
    model.load_state_dict(torch.load(""))
    
    print("Starting generating...")
    generation_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)
    model.eval()
    with torch.no_grad():
        visualize_all_validation_embeddings(model, generation_loader, DEVICE)

def visualize_region_node_embeddings(model, data_sample, graph_idx_in_batch=0):
    # model.eval()

    all_node_features = model.gnn(data_sample)

    graph_node_start_idx = data_sample.ptr[graph_idx_in_batch]
    
    region_start_idx = graph_node_start_idx + 1
    try:
        import config
        num_regions = config.NUM_ANATOMY_REGIONS
    except (ImportError, AttributeError):
        print("Warning: Cannot import config.NUM_ANATOMY_REGIONS. Using len(ANATOMY_REGION_NAMES) instead.")
        num_regions = len(INVERSE_ANA)
        
    region_end_idx = region_start_idx + num_regions
    
    region_embeddings = all_node_features[region_start_idx:region_end_idx]
    
    region_embeddings_np = region_embeddings.cpu().numpy()
    
    if region_embeddings_np.shape[0] != len(INVERSE_ANA):
        print(f"Error: Number of extracted region embeddings ({region_embeddings_np.shape[0]}) "
              f"does not match the number of region names ({len(INVERSE_ANA)}).")
        print("Please check your `ANATOMY_REGION_NAMES` list or graph structure.")
        return

    print("Running t-SNE... (this may take a moment)")
    tsne = TSNE(n_components=2, perplexity=min(10, num_regions-1), random_state=42, n_iter=1000)
    embeddings_2d = tsne.fit_transform(region_embeddings_np)
    print("t-SNE finished.")

    similarity_matrix = cosine_similarity(region_embeddings_np)

    plt.style.use('seaborn')
    fig, axes = plt.subplots(1, 2, figsize=(22, 10))
    fig.suptitle(f'Visualization of Region Node Embeddings (Graph {graph_idx_in_batch})', fontsize=20)

    ax1 = axes[0]
    scatter = ax1.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=100, c=np.arange(num_regions), cmap='viridis', alpha=0.8)
    ax1.set_title('t-SNE Visualization of Region Embeddings', fontsize=16)
    ax1.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax1.set_ylabel('t-SNE Dimension 2', fontsize=12)
    
    for i, name in enumerate(INVERSE_ANA.keys()):
        ax1.text(embeddings_2d[i, 0] + 0.1, embeddings_2d[i, 1] + 0.1, name, fontsize=9)
    
    ax2 = axes[1]
    sns.heatmap(similarity_matrix, ax=ax2, cmap='viridis', annot=False, 
                xticklabels=INVERSE_ANA.keys(), yticklabels=INVERSE_ANA.keys())
    ax2.set_title('Cosine Similarity Matrix of Region Embeddings', fontsize=16)

    ax2.tick_params(axis='x', labelsize=8)
    ax2.tick_params(axis='y', labelsize=8)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('gnn_nodes.png')
    # plt.show()

def visualize_all_validation_embeddings(model, val_dataloader, device):
    # model.eval()
    # model.to(device)

    all_region_embeddings = []
    all_region_labels = [] # e.g., [0, 1, 2, ..., 28, 0, 1, ..., 28, ...]

    try:
        import config
        num_regions = config.NUM_ANATOMY_REGIONS
    except (ImportError, AttributeError):
        print("Warning: Cannot import config.NUM_ANATOMY_REGIONS. Using len(ANATOMY_REGION_NAMES) instead.")
        num_regions = len(INVERSE_ANA.keys())

    # with torch.no_grad():
    for batch_data in tqdm(val_dataloader, desc="Extracting Validation Embeddings"):
        batch_data = batch_data.to(device)
        
        node_features_batch = model.gnn(batch_data)
        
        for i in range(batch_data.num_graphs):
            graph_node_start_idx = batch_data.ptr[i]
            region_start_idx = graph_node_start_idx + 1
            region_end_idx = region_start_idx + num_regions
            
            region_embeddings_for_one_graph = node_features_batch[region_start_idx:region_end_idx]
            
            all_region_embeddings.append(region_embeddings_for_one_graph.cpu())
            all_region_labels.extend(range(num_regions))

    if not all_region_embeddings:
        print("No embeddings were extracted. Is the validation dataloader empty?")
        return
        
    final_embeddings_tensor = torch.cat(all_region_embeddings, dim=0)
    embeddings_np = final_embeddings_tensor.numpy()
    labels_np = np.array(all_region_labels)
    
    print(f"Total number of region nodes to visualize: {embeddings_np.shape[0]}")
    print("Running t-SNE on all validation embeddings... (this might take a while)")
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000, verbose=1)
    embeddings_2d = tsne.fit_transform(embeddings_np)
    
    print("t-SNE finished.")

    plt.style.use('seaborn')
    plt.figure(figsize=(18, 14))
    
    palette = sns.color_palette("hsv", n_colors=num_regions)
    
    scatter = sns.scatterplot(
        x=embeddings_2d[:, 0], 
        y=embeddings_2d[:, 1],
        hue=[INVERSE_ANA[i] for i in labels_np],
        palette=palette,
        s=10,
        alpha=0.7,
        linewidth=0
    )
    
    plt.title('t-SNE Visualization of All Validation Region Node Embeddings', fontsize=20)
    plt.xlabel('t-SNE Dimension 1', fontsize=14)
    plt.ylabel('t-SNE Dimension 2', fontsize=14)
    
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize='small')
    
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    # plt.show()
    plt.savefig('gnn_visual_valdata_epoch20.png')

if __name__ == '__main__':
    generate()