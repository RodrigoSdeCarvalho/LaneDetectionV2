from essentials.add_module import set_working_directory
set_working_directory()

import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
from lane_detector.lane_detector import LaneDetector
from data.datasets.lane_dataset import TuSimpleDataset
from utils.path import Path
from utils.logger import Logger


def extract_lanes(frame, detector):
    Logger.trace("Processing frame", show=True)
    instances_map = detector(frame)
    return instances_map


def make_mask(instances_map):
    mask = instances_map.astype(np.uint8)
    mask = np.expand_dims(mask, axis=2)  # Add a new axis
    mask = np.repeat(mask, 3, axis=2)
    non_zero_indices = np.where(np.any(mask != [0, 0, 0], axis=-1))
    return non_zero_indices


def apply_mask(frame, mask, color=[0, 0, 255]):
    Logger.trace("Applying mask", show=True)
    for i in range(len(mask[0])):
        x, y = mask[0][i], mask[1][i]
        frame[x, y] = color
    return frame


def process_frame(frame_tensor, detector):
    # Convert tensor to image
    image = frame_tensor.cpu().numpy()
    image = np.transpose(image, (1, 2, 0))  # (H, W, 3)
    image = (image * 255) + np.array([103.939, 116.779, 123.68])
    image = np.clip(image, 0, 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Get lane instances
    instances_map = extract_lanes(image, detector)
    if instances_map is None:
        return None
    
    instances_map = instances_map.cpu().numpy()
    
    # Resize image to match model's expected size
    image = cv2.resize(image, LaneDetector.DEFAULT_IMAGE_SIZE)
    
    # Create and apply mask
    mask = make_mask(instances_map)
    image = apply_mask(image, mask)
    
    return image


def main():
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    Logger.info(f"Using device: {device}")

    # Load test dataset
    data_dir = Path().test_data
    test_set = TuSimpleDataset(data_dir, 'test')
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4)
    Logger.info(f'Data loaded from {data_dir}')

    # Initialize detector
    model_path = Path().get_model('enet-10-epochs')
    detector = LaneDetector(model_path)

    # Process each frame
    for batch in test_loader:
        frame_tensor = batch['input_tensor'][0]
        processed_frame = process_frame(frame_tensor, detector)
        
        if processed_frame is None:
            continue
            
        cv2.imshow("TuSimple Segmentation Test", processed_frame)
        key = cv2.waitKey(3000)  # 3 second delay between frames
        if key == 27:  # ESC to quit
            break
            
    cv2.destroyAllWindows()
    Logger.info("Test completed", show=True)


if __name__ == '__main__':
    main()
