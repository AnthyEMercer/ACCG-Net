import sys, os
sys.path.append(os.getcwd())

import os
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from config import ANATOMICAL_REGIONS, FINDINGS, NUM_DISEASES, NUM_ANATOMY_REGIONS, MAX_SEQ_LEN


class XRayGraphDataset(Dataset):
    def __init__(self, feature_dir, tokenizer):
        self.feature_dir = feature_dir
        self.tokenizer = tokenizer
        self.file_paths = [os.path.join(feature_dir, f) for f in os.listdir(feature_dir) if f.endswith('.pt')]

        # num_samples = 40
        # print(f"--- DEBUG MODE: Truncating dataset from {len(self.file_paths)} to {num_samples} samples. ---")
        # self.file_paths = self.file_paths[:num_samples]
        
        self._build_static_edges()

    def _build_static_edges(self):
        include_relations_str = [
            ("right lung", "right upper lung zone"), ("right lung", "right mid lung zone"),
            ("right lung", "right lower lung zone"), ("right lung", "right hilar structures"),
            ("right lung", "right apical zone"), ("right lung", "right costophrenic angle"),
            ("left lung", "left upper lung zone"), ("left lung", "left mid lung zone"),
            ("left lung", "left lower lung zone"), ("left lung", "left hilar structures"),
            ("left lung", "left apical zone"), ("left lung", "left costophrenic angle"),
            ("mediastinum", "upper mediastinum"), ("mediastinum", "aortic arch"),
            ("mediastinum", "svc"), ("mediastinum", "cardiac silhouette"),
            ("mediastinum", "right atrium"), ("mediastinum", "cavoatrial junction"),
            ("trachea", "carina"),
        ]
        INCLUDE_EDGES = [(ANATOMICAL_REGIONS[u]+1, ANATOMICAL_REGIONS[v]+1) for u, v in include_relations_str]

        related_to_relations_str = [
            ("Lung Opacity", "Consolidation"), ("Lung Opacity", "Pneumonia"),
            ("Lung Opacity", "Atelectasis"), ("Lung Opacity", "Edema"),
            ("Lung Opacity", "Lung Lesion"), ("Pleural Effusion", "Edema"),
            ("Enlarged Cardiomediastinum", "Cardiomegaly"),
        ]
        RELATED_TO_EDGES = [(FINDINGS[u]+1, FINDINGS[v]+1) for u, v in related_to_relations_str]

        self.related_to_edges = torch.tensor(
            [[u, v] for u, v in RELATED_TO_EDGES], dtype=torch.long
        ).t().contiguous()
        self.related_to_edges = torch.cat([self.related_to_edges, self.related_to_edges.flip(0)], dim=1)

        self.include_edges = torch.tensor(
            [[u, v] for u, v in INCLUDE_EDGES], dtype=torch.long
        ).t().contiguous()

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        
        data_dict = torch.load(file_path, map_location='cpu')
        global_feat = data_dict['global_feature'].unsqueeze(0) # [1, 1024]
        region_feats = data_dict['region_features']
        global_disease_probs = data_dict['global_disease_labels'].squeeze(0) # [13]
        region_disease_probs = data_dict['region_disease_labels'] # [13, 29]

        disease_indices = torch.arange(NUM_DISEASES)
        disease_feats_placeholder = torch.cat(
            [global_disease_probs.unsqueeze(1), disease_indices.unsqueeze(1).float()], dim=1
        )
    
        disease_labels = []

        has_finding_edges = []
        disease_node_start_idx = 1 + NUM_ANATOMY_REGIONS
        for region_idx in range(NUM_ANATOMY_REGIONS):
            region_disease_labels = []
            for disease_idx in range(NUM_DISEASES):
                if region_disease_probs[disease_idx, region_idx] > 0.26:
                    has_finding_edges.append([region_idx + 1, disease_idx + disease_node_start_idx])

                if region_disease_probs[disease_idx, region_idx] > 0.26:
                    region_disease_labels.append(1)
                else:
                    region_disease_labels.append(0)
            
            disease_labels.append(region_disease_labels)
        
        disease_labels_T = list(map(list, zip(*disease_labels)))
        disease_labels_T = torch.tensor(disease_labels_T, dtype=torch.float32)
        
        
        if not has_finding_edges:
            has_finding_edges = torch.empty((2, 0), dtype=torch.long)
        else:
            has_finding_edges = torch.tensor(has_finding_edges, dtype=torch.long).t().contiguous()

        root_to_region_edges = torch.tensor(
            [[0, i] for i in range(1, NUM_ANATOMY_REGIONS + 1)], dtype=torch.long
        ).t().contiguous()

        edge_index = torch.cat([root_to_region_edges, has_finding_edges, self.include_edges, self.related_to_edges], dim=1)
        edge_type = torch.cat([
            torch.full((root_to_region_edges.size(1),), 0, dtype=torch.long),
            torch.full((has_finding_edges.size(1),), 1, dtype=torch.long),
            torch.full((self.include_edges.size(1),), 2, dtype=torch.long),
            torch.full((self.related_to_edges.size(1),), 3, dtype=torch.long),
        ])
        
        total_num_nodes = 1 + NUM_ANATOMY_REGIONS + NUM_DISEASES

        # full_report_text=get_report(data_dict['dicom_id'])
        full_report_text = data_dict['report']
        
        if full_report_text is None:
            full_report_text = ""
        tokenized_output = self.tokenizer(
            full_report_text,
            padding='max_length',
            truncation=True,
            max_length=MAX_SEQ_LEN,
            return_tensors="pt"
        )
        report_input_ids = tokenized_output.input_ids.squeeze(0)
        report_attention_mask = tokenized_output.attention_mask.squeeze(0)

        region_ids = list(range(29))
        region_ids = torch.tensor(region_ids, dtype=torch.int)

        graph_data = Data(
            global_feat=global_feat,
            region_feats=region_feats,
            disease_feats=disease_feats_placeholder,
            edge_index=edge_index,
            edge_type=edge_type,
            num_nodes=total_num_nodes,
            dicom_id=data_dict['dicom_id'],
            report_input_ids=report_input_ids,
            report_attention_mask=report_attention_mask,
            region_ids = region_ids,
            region_disease = disease_labels_T
        )

        return graph_data