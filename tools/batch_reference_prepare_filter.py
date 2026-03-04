from transformers import logging
logging.set_verbosity_error()

import sys, os
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
from tqdm import tqdm
from bert_score import score as bert_score_calculator
import torch
import json
from rapidfuzz import fuzz

from config import *
from prompt import *

REGION_FILTER_1 = [2, 10, 15, 20, 23, 25, 26, 27]
REGION_FILTER_2 = [2, 3, 10, 11, 15, 16, 18, 19, 20, 23, 25, 26, 27]

INVERSE_ANA = dict((v, k) for k, v in ANATOMICAL_REGIONS.items())
ANA_NAMES = list(ANATOMICAL_REGIONS.keys())

def init_datas():
    test_file_path = ""
    p_info_file_path = "./data/patient_info.csv"
    test_df = pd.read_csv(
        test_file_path,
        usecols=['dicom_id'].extend(ANA_NAMES),
        index_col='dicom_id'
    )
    p_info_df = pd.read_csv(
        p_info_file_path,
        usecols=['dicom_id', 'info'],
        index_col='dicom_id',
        dtype={'dicom_id': str, 'info': str}
    )
    merged_df = test_df.merge(p_info_df, on='dicom_id', how='left')
    print(merged_df.head(5))
    return merged_df

def deduplicate_with_bertscore(string_list, threshold=0.95, lang="en"):
    if not string_list:
        return []

    result_list = []

    for current_string in string_list:
        is_similar_to_existing = False
        
        if not result_list:
            result_list.append(current_string)
            continue

        # cands: list of candidates
        # refs: list of references
        P, R, F1 = bert_score_calculator(
            cands=[current_string] * len(result_list),
            refs=result_list,
            lang=lang,
            model_type = "bert-base-uncased",
            verbose=False,
            device='cuda:1' if torch.cuda.is_available() else 'cpu'
        )

        for i, f1_score in enumerate(F1):
            if f1_score.item() > threshold:
                is_similar_to_existing = True
                break

        if not is_similar_to_existing:
            result_list.append(current_string)
            
    return result_list

def calculate_similarity(s1, s2):
    return fuzz.ratio(s1, s2) / 100.0

def deduplicate_similar_strings(string_list, threshold=0.85):
    result_list = []

    for current_string in string_list:
        is_similar_to_existing = False
        
        for existing_string in result_list:
            similarity = calculate_similarity(current_string, existing_string)
            if similarity > threshold:
                is_similar_to_existing = True
                break

        if not is_similar_to_existing:
            result_list.append(current_string)
            
    return result_list

if __name__ == '__main__':
    df = init_datas()

    # df = df.head(10)

    output_list = []
    pure_test = []

    for index, row in tqdm(df.iterrows()):
        region_reports = []
        for i, region in enumerate(ANA_NAMES):
            if i in REGION_FILTER_2:
                continue
            region_reports.append(row[region])
        # region_reports = deduplicate_with_bertscore(region_reports, threshold=0.95, lang="en")
        region_reports = deduplicate_similar_strings(region_reports, threshold=0.85)
        region_report_concat = ""
        region_report_only = ""
        for i, sentence in enumerate(region_reports):
            region_report_concat += f"[Region {i+1}: {sentence}] "
            region_report_only += f"{sentence} "

        prompt = prompt_template3.format(clinical_context = row['info'], image_observations = region_report_concat)
        
        output_list.append({
            "custom_id": index,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {"model": "deepseek-ai/DeepSeek-V3", "messages": [{"role":"user", "content": prompt}]},
            "max_tokens": 2048,
            "temperature": 0,
            "top_p": 0.1,
            "top_k": 1,
            "thinking_budget": 4096,
            "enable_thinking": "false"
        })
        pure_test.append([index, region_report_only.lower()])

        # print(index)
        # print(prompt)
        # input()

    # output_file = ""
    # with open(output_file, "w", encoding="utf-8") as f:
    #     for item in output_list:
    #         json_string = json.dumps(item, ensure_ascii=False)
    #         f.write(json_string + '\n')

    pure_test_file = ""
    pd.DataFrame(pure_test, columns=['dicom_id', 'Report Impression']).to_csv(pure_test_file, index=False)
    # print(f"Saved {len(output_list)} items batch file to {output_file}")
    print(f"Saved {len(pure_test)} items pure test file to {pure_test_file}")