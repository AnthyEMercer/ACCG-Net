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
            image_path = self.dataset_df.iloc[index, 0]
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            
            bbox_coordinates = self.dataset_df.iloc[index, 1] # Bounding box
            class_labels = self.dataset_df.iloc[index, 2] # Class labels (13*29 classes)
            findings = self.dataset_df.iloc[index, 4] # Finding labels (13 classes)
            region_phrases = self.dataset_df.iloc[index, 3] # Region phrases (29 regions)
            region_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
            
            transformed = self.transforms(image=image, bboxes=bbox_coordinates, region_labels=region_labels)

            transformed_image = transformed["image"]
            transformed_bboxes = transformed["bboxes"]

            sample = {
                "image": transformed_image,
                "boxes": torch.tensor(transformed_bboxes, dtype=torch.float),
                "labels": torch.tensor(class_labels, dtype=torch.int64),
                "findings": torch.tensor(findings, dtype=torch.int64),
                "region_labels": torch.tensor(region_labels, dtype=torch.int64),
                "region_phrases": region_phrases
            }
        except Exception:
            return None
        return sample
    

class CustomImageDataset_CNN(Dataset):
    def __init__(self, dataset_df, transforms):
        super().__init__()
        self.dataset_df = dataset_df
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset_df)

    def __getitem__(self, index):
        try:
            image_path = self.dataset_df.iloc[index, 0]
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            
            bbox_coordinates = self.dataset_df.iloc[index, 1] # Bounding box
            class_labels = self.dataset_df.iloc[index, 2] # Class labels (13*29 classes)
            findings = self.dataset_df.iloc[index, 4] # Finding labels (13 classes)
            region_phrases = self.dataset_df.iloc[index, 3] # Region phrases (29 regions)
            region_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
            
            transformed = self.transforms(image=image, bboxes=bbox_coordinates, region_labels=region_labels)

            transformed_image = transformed["image"]
            transformed_bboxes = transformed["bboxes"]

            dicom_id = image_path.split("/")[-1].split(".")[0]

            sample = {
                "image": transformed_image,
                "boxes": torch.tensor(transformed_bboxes, dtype=torch.float),
                "labels": torch.tensor(class_labels, dtype=torch.int64),
                "findings": torch.tensor(findings, dtype=torch.int64),
                "region_labels": torch.tensor(region_labels, dtype=torch.int64),
                "region_phrases": region_phrases,
                "dicom_id": dicom_id
            }
        except Exception:
            return None
        return sample