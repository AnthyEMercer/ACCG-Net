import sys, os
sys.path.append(os.getcwd())

import json
import re
import pandas as pd
from tqdm import tqdm

batch_response_file = ""
result_file_de = ""
result_file_concat = ""

result_f_only = ""
llm_vision = "dsv3"

def get_batch_results_de():
    contents = []
    for i in range(6):
        with open(f"{batch_response_file}_part{i+1}.jsonl", "r", encoding="utf-8") as f:
            f_bar = tqdm(f, desc=f"[Processing part{i+1}]")
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
                    repo_finding = repo_finding.replace("*", "")
                    repo_finding = repo_finding.replace("=", "")
                    repo_finding = repo_finding.replace("-", "")

                    repo_impression = repo_impression_group.group(1)
                    repo_impression = repo_impression.replace("  \n", "").replace(" \n", "").replace("\n", "")
                    
                    if repo_finding == "":
                        continue
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
    for i in range(6):
        with open(f"{batch_response_file}_part{i+1}.jsonl", "r", encoding="utf-8") as f:
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


def get_batch_result_f_only():
    contents = []
    for i in range(6):
        with open(f"{batch_response_file}_part{i+1}_{llm_vision}.jsonl", "r", encoding="utf-8") as f:
            f_bar = tqdm(f, desc=f"[Processing part{i+1}]")
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
                    if repo_finding == "":
                        continue
                    contents.append({"dicom_id": data['custom_id'], "Report Impression": repo_finding.lower()})
                    # print(contents[-1])
                except Exception as e:
                    # print(e)
                    # print(content)
                    # input()
                    continue
    df = pd.DataFrame(contents)
    df.to_csv(result_f_only, index=["dicom_id", "Report Impression"], encoding="utf-8")
    print(f"Done in {result_f_only}!")


if __name__ == "__main__":
    # get_batch_results_de()
    # get_batch_results_concat()
    get_batch_result_f_only()