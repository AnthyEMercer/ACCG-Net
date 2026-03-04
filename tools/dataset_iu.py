import cv2
import torch
from torch.utils.data import Dataset


class CustomImageDataset(Dataset):
    def __init__(self, dataset_df, transforms):
        super().__init__()
        self.dataset_df = dataset_df
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset_df)

    def __getitem__(self, index):
        try:
            image_path = self.dataset_df.iloc[index, 1]
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            transformed = self.transforms(image=image)
            transformed_image = transformed["image"]

            dicom_id = image_path.split("/")[-1].split(".")[0]
            report = self.dataset_df.iloc[index, 2]

            sample = {
                "image": transformed_image,
                "dicom_id": dicom_id,
                "report": report
            }
        except Exception:
            return None
        return sample
    

class CustomImageDataset_1(Dataset):
    def __init__(self, dataset_df, transforms):
        super().__init__()
        self.dataset_df = dataset_df
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset_df)

    def __getitem__(self, index):
        try:
            image_path = self.dataset_df.iloc[index, 1]
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            
            transformed = self.transforms(image=image)
            transformed_image = transformed["image"]

            dicom_id = image_path.split("/")[-1].split(".")[0]
            report = self.dataset_df.iloc[index, 2]

            sample = {
                "image": transformed_image,
                "dicom_id": dicom_id,
                "report": report
            }
        except Exception:
            return None
        return sample