import sys, os
sys.path.append(os.getcwd())

import json
import re
import pandas as pd
from tqdm import tqdm

batch_response_file = ""
result_file_de = ""

result_file_concat = ""

reference_file = "./data/test.csv"
reference_df = pd.read_csv(reference_file, encoding="utf-8")
reference_df = reference_df[["image_id", "reference_report"]]
reference_df.rename(columns={'image_id': 'dicom_id'}, inplace=True)


def get_batch_results_de():
    contents = []
    with open(batch_response_file, "r", encoding="utf-8") as f:
        f_bar = tqdm(f, desc="[Processing]")
        for line in f_bar:
            try:
                if not line.strip():
                    continue

                data = json.loads(line)
                content = data['response']['body']['choices'][0]['message']['content']
                repo_finding_group = re.search(r'<Finding>(.*?)</Finding>', content, re.DOTALL)
                repo_impression_group = re.search(r'<Impression>(.*?)</Impression>', content, re.DOTALL)

                repo_finding = repo_finding_group.group(1)
                repo_finding = repo_finding.replace("  \n", "").replace(" \n", "").replace("\n", "")
                repo_impression = repo_impression_group.group(1)
                repo_impression = repo_impression.replace("  \n", "").replace(" \n", "").replace("\n", "")

                contents.append({"dicom_id": data['custom_id'], "Report Impression": repo_finding.lower(), "generated_impression": repo_impression.lower()})
                # print(contents[-1])
            except Exception as e:
                # print(e)
                # print(content)
                # input()
                continue
    df = pd.DataFrame(contents)
    df.to_csv(result_file_de, index=["dicom_id", "Report Impression", "generated_impression"], encoding="utf-8")
    print(f"Done in {result_file_de}!")
    

def get_batch_results_concat():
    contents = []
    with open(batch_response_file, "r", encoding="utf-8") as f:
        f_bar = tqdm(f, desc="[Processing]")
        for line in f_bar:
            try:
                if not line.strip():
                    continue

                data = json.loads(line)
                content = data['response']['body']['choices'][0]['message']['content']
                repo_finding_group = re.search(r'<Finding>(.*?)</Finding>', content, re.DOTALL)
                repo_impression_group = re.search(r'<Impression>(.*?)</Impression>', content, re.DOTALL)

                repo_finding = repo_finding_group.group(1)
                repo_finding = repo_finding.replace("  \n", "").replace(" \n", "").replace("\n", "")
                repo_impression = repo_impression_group.group(1)
                repo_impression = repo_impression.replace("  \n", "").replace(" \n", "").replace("\n", "")

                contents.append({"dicom_id": data['custom_id'], "Report Impression": repo_finding.lower() + repo_impression.lower()})
                # print(contents[-1])
            except Exception as e:
                # print(e)
                # print(content)
                # input()
                continue
    df = pd.DataFrame(contents)
    df.to_csv(result_file_concat, index=["dicom_id", "Report Impression"], encoding="utf-8")
    print(f"Done in {result_file_concat}!")


def get_batch_noImpression():
    contents = []
    with open(batch_response_file, "r", encoding="utf-8") as f:
        f_bar = tqdm(f, desc="[Processing]")
        for line in f_bar:
            try:
                if not line.strip():
                    continue
                data = json.loads(line)
                content = data['response']['body']['choices'][0]['message']['content']
                content = content.replace("<Findings>", "<Finding>").replace("</Findings>", "</Finding>")
                repo_finding_group = re.search(r'<Finding>(.*?)</Finding>', content, re.DOTALL)

                repo_finding = repo_finding_group.group(1)
                repo_finding = repo_finding.replace("  \n", "").replace(" \n", "").replace("\n", "")
                contents.append({"dicom_id": data['custom_id'], "Report Impression": repo_finding.lower()})
                # print(contents[-1])
            except Exception as e:
                # print(e)
                # print(content)
                # input()
                continue
    df = pd.DataFrame(contents)
    result_df = pd.merge(df, reference_df, how='inner', on='dicom_id')
    result_df.to_csv(result_file_de, index=["dicom_id", "Report Impression", "reference_report"], encoding="utf-8")
    print(f"Done in {result_file_de}!")


if __name__ == "__main__":
    # get_batch_results_de()
    # get_batch_results_concat()
    get_batch_noImpression()