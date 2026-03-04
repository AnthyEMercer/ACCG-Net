import sys, os
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration
from model.GNN_model import KnowledgeGraphEncoder
import config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReportGenerator(nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.gnn = KnowledgeGraphEncoder()
        self.t5_model = T5ForConditionalGeneration.from_pretrained("")
        self.tokenizer = tokenizer
        
        model_hidden_size = self.t5_model.config.d_model
        
        self.anatomy_region_embedding = nn.Embedding(
            config.NUM_ANATOMY_REGIONS + 1, 
            model_hidden_size
        )

        NUM_DISEASE_CLASSES = 13
        self.disease_proj = nn.Linear(NUM_DISEASE_CLASSES, model_hidden_size)
        
        self.gnn_to_t5_proj = nn.Linear(config.GNN_OUTPUT_DIM, model_hidden_size)
        self.prompt_length = 3


    def _prepare_prompts(self, data):
        all_node_features = self.gnn(data)
        region_features_list = []
        for i in range(data.num_graphs):
            graph_node_start_idx = data.ptr[i]
            region_start_idx = graph_node_start_idx + 1
            region_end_idx = region_start_idx + config.NUM_ANATOMY_REGIONS
            region_features_list.append(all_node_features[region_start_idx:region_end_idx])
        gnn_features = torch.cat(region_features_list, dim=0)
        
        projected_gnn_features = self.gnn_to_t5_proj(gnn_features)
        
        anatomy_ids = data.region_ids.to(DEVICE)
        anatomy_embeds = self.anatomy_region_embedding(anatomy_ids)

        disease_logits_batch = data.region_disease.to(DEVICE)
        batch_size = data.num_graphs
        try:
            disease_logits_3d = disease_logits_batch.view(batch_size, config.NUM_DISEASES, config.NUM_ANATOMY_REGIONS)
        except RuntimeError as e:
            print(f"Error reshaping disease_logits_batch. Its shape is {disease_logits_batch.shape}")
            print(f"Expected shape to be compatible with (batch_size={batch_size}, num_diseases={config.NUM_DISEASES}, num_regions={config.NUM_ANATOMY_REGIONS})")
            raise e
        
        disease_logits_transposed = disease_logits_3d.transpose(1, 2)
        disease_logits_reshaped = disease_logits_transposed.reshape(-1, config.NUM_DISEASES)

        disease_embeds = self.disease_proj(disease_logits_reshaped)
        
        prompt_embeds = torch.cat([
            anatomy_embeds.unsqueeze(1),
            disease_embeds.unsqueeze(1),
            projected_gnn_features.unsqueeze(1)
        ], dim=1)
        
        # [B * R, 3, hidden_size]
        return prompt_embeds
    
    def forward(self, data):
        # prompt_embeds [B * R, 3, hidden_size]
        prompt_embeds = self._prepare_prompts(data)
        labels = data.region_labels
        
        prompt_attention_mask = torch.ones(
            prompt_embeds.size(0), self.prompt_length, device=prompt_embeds.device
        )
        attention_mask = prompt_attention_mask
        
        outputs = self.t5_model(
            inputs_embeds=prompt_embeds,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs.loss

    @torch.no_grad()
    def generate(self, data, max_length=50, num_beams=4, early_stopping=True):
        self.eval()
        prompt_embeds = self._prepare_prompts(data)
        
        attention_mask = torch.ones(
            prompt_embeds.size(0), self.prompt_length, device=prompt_embeds.device
        )
        
        generated_ids = self.t5_model.generate(
            inputs_embeds=prompt_embeds,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=early_stopping,
        )
        return generated_ids