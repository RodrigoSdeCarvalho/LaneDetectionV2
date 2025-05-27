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

# Color palette for up to 10 lanes
LANE_COLORS = [
    (60, 76, 231), (18, 156, 243), (113, 204, 46), (219, 152, 52), (182, 89, 155),
    (94, 73, 52), (0, 84, 211), (15, 196, 241), (156, 188, 26), (185, 128, 41)
]

def sample_from_curve(curves, inst_pred, y_sample):
    h, w = inst_pred.shape
    curves_pts = []
    for param in curves:
        fy = np.poly1d(param)
        x_sample = fy(y_sample)
        # Filter out-of-bounds
        index = np.where(np.logical_or(x_sample < 0, x_sample >= w))
        x_sample[index] = -2
        # Filter out points not in the prediction
        mask = (x_sample != -2)
        x_int = np.round(x_sample[mask]).astype(np.int32)
        y_int = y_sample[mask]
        valid = (inst_pred[y_int, x_int] != 0)
        x_sample[mask][~valid] = -2
        xy_sample = np.vstack((x_sample, y_sample)).transpose((1, 0)).astype(np.int32)
        curves_pts.append(xy_sample)
    return curves_pts

def process_frame(frame_tensor, detector):
    image = frame_tensor.cpu().numpy()
    image = np.transpose(image, (1, 2, 0))  # (H, W, 3)
    image = (image * 255) + np.array([103.939, 116.779, 123.68])
    image = np.clip(image, 0, 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    instances_map = detector(image)
    instances_map_np = instances_map.cpu().numpy() if torch.is_tensor(instances_map) else instances_map
    h, w = instances_map_np.shape

    # Use degree=3 for curve fitting
    curves = detector.fit_lanes(instances_map_np)
    
    # Sample points along the y-axis
    y_sample = np.linspace(160 * h / 720, 710 * h / 720, 56, dtype=np.int16)
    curves_pts = detector.sample_from_curve(curves, instances_map, y_sample)

    # Draw the fitted curves
    for idx, curve_pts in enumerate(curves_pts):
        color = LANE_COLORS[idx % len(LANE_COLORS)]
        for pt in curve_pts:
            x, y = pt
            if x >= 0:  # Only draw valid points
                cv2.circle(image, (x, y), 2, color, -1)

    return image

def main():
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    Logger.info(f"Using device: {device}")

    data_dir = Path().test_data
    test_set = TuSimpleDataset(data_dir, 'test')
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4)
    Logger.info(f'Data loaded from {data_dir}')

    model_path = Path().get_model('enet-10-epochs')
    detector = LaneDetector(model_path)

    for batch in test_loader:
        frame_tensor = batch['input_tensor'][0]
        processed_frame = process_frame(frame_tensor, detector)
        cv2.imshow("TuSimple Fitted Lanes Test", processed_frame)
        key = cv2.waitKey(3000)
        if key == 27:  # ESC to quit
            break
    cv2.destroyAllWindows()
    Logger.info("Test completed", show=True)

if __name__ == '__main__':
    main()
