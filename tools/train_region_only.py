import sys, os
sys.path.append(os.getcwd())

import torch
from torch.optim import AdamW
from torch_geometric.loader import DataLoader
from transformers import T5Tokenizer
from tqdm import tqdm
import itertools

import config
from tools.KG.KG_Dataset import XRayGraphDataset
from model.Generator_region_only import ReportGenerator

INVERSE_ANA = dict((v, k) for k, v in config.ANATOMICAL_REGIONS.items())

TRAIN_NODE_FILE = "./data/node_feat_train"
VAL_NODE_FILE = "./data/node_feat_val"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    tokenizer = T5Tokenizer.from_pretrained("")
    
    train_dataset = XRayGraphDataset(TRAIN_NODE_FILE, tokenizer)
    val_dataset = XRayGraphDataset(VAL_NODE_FILE, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    model = ReportGenerator(tokenizer).to(DEVICE)
    # model.load_state_dict(torch.load(""))
    
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
    
    print("Starting training for region-specific generation...")
    best_val_loss = float('inf')
    
    for epoch in range(config.NUM_EPOCHS):
        # if epoch < 20:
        #     continue
        model.train()
        total_train_loss = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
        for batch in train_bar:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            loss = model(batch)  # model.forward()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            train_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")
        with torch.no_grad():
            for batch in val_bar:
                batch = batch.to(DEVICE)
                loss = model(batch)
                total_val_loss += loss.item()
                val_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")

        print("\n--- Generating sample sentences for validation data ---")
        model.eval()
        
        generation_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)
        
        with torch.no_grad():
            sample_batch = next(iter(generation_loader))
            sample_batch = sample_batch.to(DEVICE)

            generated_ids = model.generate(sample_batch)
            generated_sentences = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            ground_truth_ids = sample_batch.region_labels
            ground_truth_ids[ground_truth_ids == -100] = tokenizer.pad_token_id
            ground_truth_sentences = tokenizer.batch_decode(ground_truth_ids, skip_special_tokens=True)
            
            print(f"\n----- Displaying results for a sample graph from the batch -----")
            for i in range(config.NUM_ANATOMY_REGIONS):
                gt_text = ground_truth_sentences[i]
                gen_text = generated_sentences[i]
                
                if gt_text: 
                    print(f"  Region {i+1} ({INVERSE_ANA[i]}):")
                    print(f"    - Ground Truth: {gt_text}")
                    print(f"    - Generated   : {gen_text}")
            print("----------------------------------------------------------------")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = f'./model/run_region_only/region_model_epoch{epoch+1}_loss-{avg_val_loss:.4f}.pth'
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved to {save_path}")

    print("Training complete.")

if __name__ == '__main__':
    main()