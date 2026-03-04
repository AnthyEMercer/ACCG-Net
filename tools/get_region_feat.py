import sys, os
sys.path.append(os.getcwd())

import torch
import torchvision.transforms as T
from PIL import Image

from tools.object_detector.object_detector import ObjectDetector
from config import IMG_SIZE_FRC
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import List, Dict, Tuple, Any, Optional
import cv2

MODEL_WEIGHTS_PATH = ""
# EXAMPLE_IMAGE_PATH = ""
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Instantiating Faster R-CNN model...")
model = ObjectDetector(return_feature_vectors=True)
model.to(DEVICE)

print(f"Loading weights from {MODEL_WEIGHTS_PATH}...")
try:
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE))
    print("Weights loaded successfully.")
except FileNotFoundError:
    print(f"ERROR: Weight file not found at {MODEL_WEIGHTS_PATH}. Running with initialized weights for demonstration.")
except Exception as e:
    print(f"ERROR: Could not load weights: {e}. Running with initialized weights for demonstration.")

model.eval()
print("Model set to evaluation mode.")

def get_anatomical_region_data(
    image_batch_tensor: torch.Tensor
) -> Optional[List[List[Dict[str, Any]]]]:
    if not isinstance(model, torch.nn.Module):
        print("Error: Invalid model provided.")
        return None
    if not isinstance(image_batch_tensor, torch.Tensor) or image_batch_tensor.ndim != 4:
        print(f"Error: Invalid image_batch_tensor. Expected shape [N, C, H, W], got {image_batch_tensor.shape if isinstance(image_batch_tensor, torch.Tensor) else type(image_batch_tensor)}")
        return None

    model.eval() # Ensure model is in evaluation mode

    try:
        with torch.no_grad():
            _, detections_dict, top_region_features_batch, class_detected_batch = model(images=image_batch_tensor, targets=None)

        batch_size = image_batch_tensor.shape[0]
        num_anatomical_regions = model.num_classes - 1

        if not (isinstance(detections_dict, dict) and
                'top_region_boxes' in detections_dict and
                'top_scores' in detections_dict):
            print("Error: Model output 'detections' dictionary is not in the expected format.")
            print(f"Received detections_dict: {detections_dict}")
            return None

        top_boxes_batch = detections_dict['top_region_boxes']   # Expected shape [N, 29, 4]
        top_scores_batch = detections_dict['top_scores']     # Expected shape [N, 29]

        # --- Sanity checks for batch shapes ---
        expected_shape_feat_batch = (batch_size, num_anatomical_regions, top_region_features_batch.shape[-1])
        expected_shape_box_batch = (batch_size, num_anatomical_regions, 4)
        expected_shape_score_batch = (batch_size, num_anatomical_regions)
        expected_shape_detected_batch = (batch_size, num_anatomical_regions)

        if top_region_features_batch.shape != expected_shape_feat_batch:
            print(f"Warning: top_region_features_batch shape mismatch. Expected {expected_shape_feat_batch}, got {top_region_features_batch.shape}")
        if top_boxes_batch.shape != expected_shape_box_batch:
            print(f"Warning: top_boxes_batch shape mismatch. Expected {expected_shape_box_batch}, got {top_boxes_batch.shape}")
        if top_scores_batch.shape != expected_shape_score_batch:
            print(f"Warning: top_scores_batch shape mismatch. Expected {expected_shape_score_batch}, got {top_scores_batch.shape}")
        if class_detected_batch.shape != expected_shape_detected_batch:
            print(f"Warning: class_detected_batch shape mismatch. Expected {expected_shape_detected_batch}, got {class_detected_batch.shape}")
        # --- End Sanity checks ---

        all_images_region_data = []
        for i in range(batch_size): # Iterate over each image in the batch
            # Extract data for the i-th image
            top_region_features_img = top_region_features_batch[i] # [29, 1024]
            top_boxes_img = top_boxes_batch[i]                     # [29, 4]
            top_scores_img = top_scores_batch[i]                   # [29]
            class_detected_img = class_detected_batch[i]           # [29] (boolean)

            processed_regions_for_img = []
            for region_idx in range(num_anatomical_regions):
                region_data = {
                    'region_id': region_idx + 1,
                    'feature_vector': top_region_features_img[region_idx].cpu(),
                    'bounding_box': top_boxes_img[region_idx].cpu(),
                    'score': top_scores_img[region_idx].item(),
                    'is_detected': class_detected_img[region_idx].item()
                }
                processed_regions_for_img.append(region_data)
            all_images_region_data.append(processed_regions_for_img)

        return all_images_region_data

    except Exception as e:
        print(f"Error during model inference or data processing: {e}")
        import traceback
        traceback.print_exc()
        return None