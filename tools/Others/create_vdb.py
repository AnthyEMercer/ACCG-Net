from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.document_loaders import CSVLoader
from langchain.schema import Document
import pandas as pd
import sys, os
from tqdm import tqdm

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import urllib3
import ssl
ssl._create_default_https_context = ssl._create_unverified_context()

http = urllib3.PoolManager(
    cert_reqs = 'CERT_NONE'
)
import requests
s = requests.Session()
s.verify = False

CLASS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum", "Fracture",
         "Lung Lesion", "Lung Opacity", "Pleural Effusion", "Pleural Other", "Pneumonia",
         "Pneumothorax", "Support Devices"]
RAG_CKPT = "./model/bge-m3"
TRAIN_PATH = './data/train_chexbert.csv'
TEST_PATH = './data/test_chexbert.csv'
VAL_PATH = './data/val_chexbert.csv'
SOURCE_INFO_PATH = "./data/patient_info.csv"
VDB_INFO_PATH = "./data/vdb/mimic"
VDB_CHROMA_PATH = "./data/vdb/mimic_chroma"

def create_db(list_of_documents, db_path):
    print("> Initializing embeddings model...")
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=RAG_CKPT,
        model_kwargs={'device': 'cuda:0'}
    )
    print("> Creating database from documents...")
    # db = FAISS.from_documents(list_of_documents, embeddings)
    # db.save_local(db_path)
    db = Chroma.from_documents(
        documents=list_of_documents, 
        embedding=embeddings, 
        persist_directory=db_path
    )
    print(f"> Finish creating database! Saved in {db_path}")
    print(f"> Total documents in the new database: {db._collection.count()}")

def prepare_documents_with_metadata():
    print("> Reading and processing source CSV files...")
    df = pd.read_csv(TRAIN_PATH, header=0)
    df['study_id'] = df['study_id'].astype(str)
    df.drop_duplicates(subset='study_id', keep='first', inplace=True)
    df.set_index(["dicom_id"])
    df.fillna(0, inplace=True)
    df.replace(-1.0, 1.0, inplace=True)

    info_df = pd.read_csv(SOURCE_INFO_PATH, header=0)
    info_df['study_id'] = info_df['study_id'].astype(str)
    info_df.set_index(["dicom_id"])
    info_df.fillna(0, inplace=True)
    info_df.replace(-1.0, 1.0, inplace=True)

    list_of_documents = []
    print("> Preparing documents with content and metadata...")
    skipped_count = 0
    for idx in tqdm(df.index, position=0):
        line = df.loc[idx]
        report = line['Report Impression']
        if pd.isna(report) or str(report) == '-1':
            continue
        
        if not report or report in [".", ",", ";", "0"]:
            skipped_count += 1
            continue
        
        MIN_CHAR_LENGTH = 40
        MIN_WORD_COUNT = 10

        if len(report) < MIN_CHAR_LENGTH or len(report.split()) < MIN_WORD_COUNT:
            skipped_count += 1
            continue

        try:
            info = info_df.loc[idx]['info']
        except KeyError:
            info = "No additional patient info available."

        p_id = line['study_id']
        
        label_string = ""
        for pa in CLASS[:-1]:
            if line[pa] == 1:
                label_string += pa + '.'
        if label_string == "":
            label_string = "No lesions present."
        if line[CLASS[-1]] == 1:
            label_string += CLASS[-1] + '.'
        
        page_content = f"Patient Info: [{info}]. Report: [{report}]"

        doc_metadata = {"case_label": label_string, "study_id": p_id}
        
        doc = Document(page_content=page_content, metadata=doc_metadata)
        list_of_documents.append(doc)
    
    print(f"> Finished preparing {len(list_of_documents)} documents.")
    print(f"> Skipped {skipped_count} documents due to insufficient length or invalid content.")
    return list_of_documents

if __name__ == '__main__':
    documents = prepare_documents_with_metadata()
    
    if documents:
        print("> Start to create database...")
        create_db(documents, VDB_CHROMA_PATH)
        print("> Done.")
    else:
        print("> No documents were prepared. Database creation skipped.")