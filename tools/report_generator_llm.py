import sys, os
sys.path.append(os.getcwd())

from openai import OpenAI
import pandas as pd
from tqdm import tqdm
import re

from config import OPENROUTER, ANATOMICAL_REGIONS
from prompt import *

INVERSE_ANA = dict((v, k) for k, v in ANATOMICAL_REGIONS.items())
ANA_NAMES = list(ANATOMICAL_REGIONS.keys())

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER,
)

def init_datas():
    test_file_path = "./reports/test_outputs_1.csv"
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

def test_LLM():
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER,
    )

    completion = client.chat.completions.create(
        extra_body={},
        model="deepseek/deepseek-chat-v3-0324:free",
        messages=[
            {
            "role": "user",
            "content": "Whats your name'?"
            }
        ]
    )
    print(completion.choices[0].message.content)

def llm(c_context, region_report):
    completion = client.chat.completions.create(
        extra_body={},
        model="deepseek/deepseek-chat-v3-0324:free",
        messages=[
            {
            "role": "user",
            "content": prompt_template1.format(clinical_context=c_context, image_observations=region_report)
            }
        ]
    )
    # print(f"Prompt: {prompt_template1.format(clinical_context=c_context, image_observations=region_report)}")
    # print(completion.choices[0].message.content)
    # input()
    return completion.choices[0].message.content

if __name__ == "__main__":
    data_df = init_datas()
    final_report = []
    for index, row in tqdm(data_df.iterrows()):
        region_reports = ""
        for i, region in enumerate(ANA_NAMES):
            region_reports += f"Region:{i}: {row[region]}\n"
        # print(region_reports)
        try:
            llm_response = llm(row['info'], region_reports)
            repo_finding = re.search(r'<Finding>(.*?)</Finding>', llm_response, re.DOTALL)
            repo_impression = re.search(r'<Impression>(.*?)</Impression>', llm_response, re.DOTALL)
            final_report.append({
                'dicom_id': index,
                'finding': repo_finding.group(1),
                'impression': repo_impression.group(1),
            })
            # print(final_report[0])
            # input()
        except Exception as e:
            print(e)
            print(f"Error return reports: {llm_response}")
            continue


    final_df = pd.DataFrame(final_report)
    final_df.to_csv("./reports/final_reports_1.csv", index=['dicom_id', 'finding', 'impression'])