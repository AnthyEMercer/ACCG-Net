import sys, os
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGATConv
import config

class KnowledgeGraphEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.disease_embedding = nn.Embedding(config.NUM_DISEASES, config.DISEASE_NAME_EMBED_DIM)
        self.global_proj = nn.Linear(config.GLOBAL_FEAT_DIM, config.GNN_HIDDEN_DIM)
        self.region_proj = nn.Linear(config.REGION_FEAT_DIM, config.GNN_HIDDEN_DIM)
        
        self.disease_proj = nn.Linear(1 + config.DISEASE_NAME_EMBED_DIM, config.GNN_HIDDEN_DIM)

        # self.conv1 = RGATConv(config.GNN_HIDDEN_DIM, config.GNN_OUTPUT_DIM, num_relations=4, heads=1)

        self.conv1 = RGATConv(config.GNN_HIDDEN_DIM, config.GNN_HIDDEN_DIM, num_relations=4, heads=4)
        self.conv2 = RGATConv(config.GNN_HIDDEN_DIM * 4, config.GNN_OUTPUT_DIM, num_relations=4, heads=1)

        
        # self.conv1 = RGATConv(config.GNN_HIDDEN_DIM, config.GNN_HIDDEN_DIM, num_relations=4, heads=4)
        # self.conv2 = RGATConv(config.GNN_HIDDEN_DIM * 4, config.GNN_OUTPUT_DIM, num_relations=4, heads=4)
        # self.conv3 = RGATConv(config.GNN_OUTPUT_DIM * 4, config.GNN_OUTPUT_DIM, num_relations=4, heads=1)

    def forward(self, data):
        global_feat = self.global_proj(data.global_feat)
        region_feats = self.region_proj(data.region_feats)
        disease_probs = data.disease_feats[:, 0].unsqueeze(1)
        disease_indices = data.disease_feats[:, 1].long()
        disease_name_embeds = self.disease_embedding(disease_indices)
        combined_disease_feats = torch.cat([disease_probs, disease_name_embeds], dim=1)
        disease_feats = self.disease_proj(combined_disease_feats)
        
        x = torch.cat([global_feat, region_feats, disease_feats], dim=0)
        
        # x = self.conv1(x, data.edge_index, data.edge_type)

        x = F.relu(self.conv1(x, data.edge_index, data.edge_type))
        x = self.conv2(x, data.edge_index, data.edge_type)
        
        # x = F.relu(self.conv1(x, data.edge_index, data.edge_type))
        # x = F.relu(self.conv2(x, data.edge_index, data.edge_type))
        # x = self.conv3(x, data.edge_index, data.edge_type)
        return x