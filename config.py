val_dataset_path = "./data/val"
train_dataset_path = "./data/train"
test_dataset_path = "./data/test"

NUM_ANATOMY_REGIONS = 29
NUM_DISEASES = 13

IMG_SIZE_FRC = 512
batch_size = 3
CKPT_S2 = ""
CKPT_FRC = ""
LEARNING_RATE = 1e-4
NUM_EPOCHS = 20
# GNN
GNN_HIDDEN_DIM = 256
GNN_OUTPUT_DIM = 256
# GPT2
MAX_SEQ_LEN = 128
GPT2_MODEL = ''
GPT2_EMBED_DIM = 512

GLOBAL_FEAT_DIM = 1024
REGION_FEAT_DIM = 1024
DISEASE_PROB_DIM = 13
DISEASE_NAME_EMBED_DIM = 128
DISEASE_FEAT_DIM = 1 + DISEASE_NAME_EMBED_DIM

NUM_WORKERS = 4 # DataLoader num_workers
RANDOM_SEED = 42

# Region settings
ANATOMICAL_REGIONS = {
    "right lung": 0,
    "right upper lung zone": 1,
    "right mid lung zone": 2,
    "right lower lung zone": 3,
    "right hilar structures": 4,
    "right apical zone": 5,
    "right costophrenic angle": 6,
    "right hemidiaphragm": 7,
    "left lung": 8,
    "left upper lung zone": 9,
    "left mid lung zone": 10,
    "left lower lung zone": 11,
    "left hilar structures": 12,
    "left apical zone": 13,
    "left costophrenic angle": 14,
    "left hemidiaphragm": 15,
    "trachea": 16,
    "spine": 17,
    "right clavicle": 18,
    "left clavicle": 19,
    "aortic arch": 20,
    "mediastinum": 21,
    "upper mediastinum": 22,
    "svc": 23,
    "cardiac silhouette": 24,
    "cavoatrial junction": 25,
    "right atrium": 26,
    "carina": 27,
    "abdomen": 28
}

FINDINGS = {
    "Atelectasis": 29, 
    "Cardiomegaly": 30, 
    "Consolidation": 31, 
    "Edema": 32, 
    "Enlarged Cardiomediastinum": 33, 
    "Fracture": 34,
    "Lung Lesion": 35, 
    "Lung Opacity": 36, 
    "Pleural Effusion": 37, 
    "Pleural Other": 38,
    "Pneumonia": 39,
    "Pneumothorax": 40, 
    "Support Devices": 41
}

# LLM api-kay
OPENROUTER = ""
SILICONFLOW = ""