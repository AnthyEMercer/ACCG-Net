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
import csv
import pandas as pd
import time

import config
from tools.KG.KG_Dataset_iu import XRayGraphDataset
from model.Generator_3p import ReportGenerator
# from model.Generator_region_only import ReportGenerator
TRAIN_NODE_FILE = "./data/iu_node_feat_train"
VAL_NODE_FILE = "./data/iu_node_feat_val"
TEST_NODE_FILE = "./data/iu_node_feat_test"

INVERSE_ANA = dict((v, k) for k, v in config.ANATOMICAL_REGIONS.items())

DEVICE = torch.device("cuda:0")

def generate():
    tokenizer = T5Tokenizer.from_pretrained("")
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token

    test_dataset = XRayGraphDataset(TEST_NODE_FILE, tokenizer)

    model = ReportGenerator(tokenizer).to(DEVICE)
    model.load_state_dict(torch.load(""))
    
    print("Starting generating...")
    generation_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    model.eval()

    outputs = []
    test_bar = tqdm(generation_loader, desc=f"[Test]")

    with torch.no_grad():
        for i, sample_batch in enumerate(test_bar):
            # dicom_id
            dicom_ids = sample_batch.dicom_id

            sample_batch = sample_batch.to(DEVICE)
            generated_ids = model.generate(sample_batch)
            generated_reports = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            for i, dicom_id in enumerate(dicom_ids):
                output = {}
                output["dicom_id"] = dicom_id
                for j in range(config.NUM_ANATOMY_REGIONS):
                    output[INVERSE_ANA[j]] = generated_reports[i * config.NUM_ANATOMY_REGIONS + j]
                outputs.append(output)
            
    outdf = pd.DataFrame(outputs)
    out_file_path = f"test_outputs_iu_{int(time.time())}.csv"
    outdf.to_csv(out_file_path, index=["dicom_id"] + list(INVERSE_ANA.values()), encoding="utf-8")
    print(f"Done in {out_file_path}!")
                

if __name__ == "__main__":
    generate()
    # generate_single()