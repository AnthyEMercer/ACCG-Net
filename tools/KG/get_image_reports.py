import sys, os
sys.path.append(os.getcwd())

import os
import pandas as pd

import config

test_file = "./data/test.csv"
# train_file = "./data/train.csv"
val_file = "./data/valid.csv"
train_with_report_file = "./data/train_with_report.csv"

loaded_dfs = []
for path in [test_file, val_file]:
    df = pd.read_csv(
        path,
        usecols=['image_id', 'reference_report'],
        dtype={'image_id': str, 'reference_report': str}
    )
    df.rename(columns={"image_id": 'dicom_id'}, inplace=True)
    df.set_index('dicom_id', inplace=True)
    loaded_dfs.append(df)

df = pd.read_csv(
    train_with_report_file,
    usecols=['dicom_id', 'reference_report'],
    index_col='dicom_id',
    dtype={'dicom_id': str, 'reference_report': str}
)
loaded_dfs.append(df)

def get_report(dicom_id):
    for df in loaded_dfs:
        try:
            report = df.loc[dicom_id, 'reference_report']
            return report
        except KeyError:
            continue
            
    return None