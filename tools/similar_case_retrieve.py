from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS, Chroma

RAG_CKPT = ""
VDB_INFO_PATH = ""
VDB_INFO_METADATA_PATH = ""
VDB_CHROMA_PATH = ""

CLASS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum", "Fracture",
         "Lung Lesion", "Lung Opacity", "Pleural Effusion", "Pleural Other", "Pneumonia",
         "Pneumothorax", "Support Devices"]


persist_directory = VDB_CHROMA_PATH
# db = FAISS.load_local(persist_directory, HuggingFaceBgeEmbeddings(model_name=RAG_CKPT, model_kwargs={'device': 'cpu'}), allow_dangerous_deserialization=True)
db = Chroma(persist_directory=persist_directory, embedding_function=HuggingFaceBgeEmbeddings(model_name=RAG_CKPT, model_kwargs={'device': 'cpu'}))

def similar_case_retrieve_922(pa_info, label_line):
    label = ""
    for cls in CLASS[:-1]:
        if label_line[cls] == 1:
            label += cls + '.'
    if label == "":
        label = "No lesions present."
    if label_line[CLASS[-1]] == 1:
        label += CLASS[-1] + '.'

    # print(f"---Query: \n>Patient_info: {pa_info}, \n>Label:{label}")
    query = f"Which patient has the similarist situation as following: {pa_info}. Radiological manifestations: {label}"

    k = 3
    retriever = db.similarity_search(query, k=k)
    examples = ""

    for i in range(k):
        # print(f"---Retrieve case[{i}] : {retriever[i].page_content}")
        content = retriever[i].page_content.split('REPORT:')[1]
        content = content.split(']')[0]
        examples += f"Example {i+1}: {content}\n"

    return examples


def similar_case_retrieve(pa_info, label_line):
    label_to_filter = ""
    for cls in CLASS[:-1]:
        if label_line[cls] == 1:
            label_to_filter += cls + '.'
    if label_to_filter == "":
        label_to_filter = "No lesions present."
    if label_line[CLASS[-1]] == 1:
        label_to_filter += CLASS[-1] + '.'

    metadata_filter = {"case_label": label_to_filter}

    query = f"Patient Info: [{pa_info}]."
    k = 3
    
    # db = FAISS.load_local(VDB_INFO_PATH, embeddings_model)
    retrieved_docs = db.similarity_search(query, k=k, filter=metadata_filter)
    
    examples = ""
    # print(f"---Query: \n>Patient_info: {query}, \n>Filter: {metadata_filter}")
    
    cnt_e = 0
    for i, doc in enumerate(retrieved_docs):
        # print(f"---Retrieve case[{i}] : {doc.page_content}")
        # print(f"---Metadata : {doc.metadata}")
        content = doc.page_content.split('Report:')[1]
        if len(content) <= 100:
            continue
        else:
            examples += f"Example {cnt_e+1}: {content.strip()}\n"
            cnt_e += 1

    return examples