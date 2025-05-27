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


def process_frame(frame_tensor, detector):
    # Convert tensor to image
    image = frame_tensor.cpu().numpy()
    image = np.transpose(image, (1, 2, 0))  # (H, W, 3)
    image = (image * 255) + np.array([103.939, 116.779, 123.68])
    image = np.clip(image, 0, 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Get lane instances
    instances_map = detector(image)
    if instances_map is None:
        return image
    
    # Get center of lane points
    center_points = detector.get_center_of_lane(instances_map, get_map=False)
    if center_points is None:
        return image

    # Draw the center line as a dotted line
    for i, pt in enumerate(center_points):
        if i % 5 == 0:  # Draw every 5th point for a dotted effect
            x, y = int(round(pt[0])), int(round(pt[1]))
            if x >= 0 and y >= 0 and x < image.shape[1] and y < image.shape[0]:
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)  # Green center line

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
        
        cv2.imshow("TuSimple Center of Lane Test", processed_frame)
        key = cv2.waitKey(3000)  # 3 second delay between frames
        if key == 27:  # ESC to quit
            break
            
    cv2.destroyAllWindows()
    Logger.info("Test completed", show=True)


if __name__ == '__main__':
    main()
