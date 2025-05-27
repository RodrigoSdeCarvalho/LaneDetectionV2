from lane_detector.lane import Lanes
from lane_detector.memory import Memory
from neural_networks.enet import Enet
import torch
from sklearn.cluster import MeanShift
import cv2
import numpy as np
from typing import Optional
from utils.logger import Logger
from typing import List
import warnings
from utils.config import ImageConfig, DecoratorConfig


warnings.filterwarnings("ignore")


# Note, in numpy.where, the first index is the y-axis (0), and the second index is the x-axis (1).
class LaneDetector:
    DEFAULT_IMAGE_SIZE = (ImageConfig().preprocess_width, ImageConfig().preprocess_height)
    LANE_POINT_MASK = 1
    CENTER_POINT_MASK = 2
    VGG_MEAN = np.array([103.939, 116.779, 123.68]).astype(np.float32)

    def __init__(self, model_path: str):
        # Initialize model
        self._model = Enet()
        self._model = torch.nn.DataParallel(self._model)
        
        # Setup device
        self._device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self._model.to(self._device)
        
        # Load model if path provided
        if model_path:
            checkpoint = torch.load(model_path)
            self._model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        else:
            raise Exception("Model path must be provided")
            
        self._model.eval()
        
        self._memory = Memory()

    def __call__(self, image) -> Optional[torch.Tensor]:
        image = self._preprocess_image(image)
        with torch.no_grad():
            embeddings, logit = self._model(image)
        pred_bin = torch.argmax(logit, dim=1, keepdim=True)  # (1, 1, H, W)
        instance_map = self._cluster_embed(embeddings, pred_bin)
        if instance_map is None:
            return self._memory.get_instances_map()
        self._memory.update(torch.from_numpy(instance_map))\

        return torch.from_numpy(instance_map)

    def _preprocess_image(self, image):
        image = cv2.resize(image, self.DEFAULT_IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        image = image.astype(np.float32) / 255.0  # Normalize to [0,1]
        image = torch.from_numpy(image).float().permute((2, 0, 1)).unsqueeze(dim=0).to(self._device)

        return image

    def _cluster_embed(self, embeddings, pred_bin, band_width=0.5):
        c = embeddings.shape[1]
        n, _, h, w = pred_bin.shape
        preds_bin = pred_bin.view(n, h, w)
        preds_inst = torch.zeros_like(preds_bin)
        for idx, (embedding, bin_pred) in enumerate(zip(embeddings, preds_bin)):
            bin_pred_bool = bin_pred.bool()
            embedding_fg = torch.transpose(torch.masked_select(embedding, bin_pred_bool).view(c, -1), 0, 1)
            if embedding_fg.shape[0] == 0:
                continue
            clustering = MeanShift(bandwidth=band_width, bin_seeding=True, min_bin_freq=100).fit(embedding_fg.cpu().detach().numpy())
            labels = torch.from_numpy(clustering.labels_).to(embedding.device) + 1
            preds_inst[idx][bin_pred_bool] = labels
        return preds_inst[0].cpu().numpy()  # single image

    def get_center_of_lane(self, instances_map, get_map: bool = False) -> Optional[np.ndarray]:
        h, w = instances_map.shape
        y_sample = np.arange(h)
        curves = self.fit_lanes(instances_map)
        if not curves or len(curves) < 2:
            cached_lanes = self._memory.get_lanes()
            if cached_lanes is None:
                return None
            curves = cached_lanes

        lane_xs = []
        for curve in curves:
            fy = np.poly1d(curve)
            x_sample = fy(y_sample)
            lane_xs.append(x_sample)
        lane_xs = np.array(lane_xs)  # shape: (num_lanes, h)

        # For each lane, get its mean x position (over valid y)
        lane_means = [np.nanmean(xs[(xs >= 0) & (xs < w)]) for xs in lane_xs]
        center_x_img = w / 2
        # Left: closest mean x < center; Right: closest mean x > center
        left_idx = None
        right_idx = None
        min_left_dist = float('inf')
        min_right_dist = float('inf')
        for i, mean_x in enumerate(lane_means):
            if np.isnan(mean_x):
                continue
            dist = abs(mean_x - center_x_img)
            if mean_x < center_x_img and dist < min_left_dist:
                left_idx = i
                min_left_dist = dist
            if mean_x > center_x_img and dist < min_right_dist:
                right_idx = i
                min_right_dist = dist
        if left_idx is None or right_idx is None:
            return None
        left_xs = lane_xs[left_idx]
        right_xs = lane_xs[right_idx]

        # Get the y-values used for fitting each curve (from fit_lanes)
        inst_pred = torch.from_numpy(instances_map) if not torch.is_tensor(instances_map) else instances_map
        inst_pred_expand = inst_pred.view(-1)
        inst_unique = torch.unique(inst_pred_expand)
        lanes_pts = []
        for inst_idx in inst_unique:
            if inst_idx != 0:
                lanes_pts.append(torch.nonzero(inst_pred == inst_idx).cpu().numpy())
        if len(lanes_pts) < 2:
            return None
        left_pts = lanes_pts[left_idx]
        right_pts = lanes_pts[right_idx]
        # y-values are in the first column (row index)
        min_y = min(left_pts[:, 0].min(), right_pts[:, 0].min())
        max_y = max(left_pts[:, 0].max(), right_pts[:, 0].max())
        # Restrict y_sample to this range
        y_range_mask = (y_sample >= min_y) & (y_sample <= max_y)
        y_sample_range = y_sample[y_range_mask]
        left_xs_range = left_xs[y_range_mask]
        right_xs_range = right_xs[y_range_mask]
        # Only use y where both left and right are valid
        valid_mask = (left_xs_range >= 0) & (left_xs_range < w) & (right_xs_range >= 0) & (right_xs_range < w)
        center_x = (left_xs_range[valid_mask] + right_xs_range[valid_mask]) / 2
        y_valid = y_sample_range[valid_mask]
        if len(center_x) == 0:
            return None
        center_points = np.stack([center_x, y_valid], axis=1)

        self._memory.update(center_points)
        Logger.trace(f"Center points: {center_points}", show=True)

        if get_map:
            if torch.is_tensor(instances_map):
                instances_map = instances_map.cpu().numpy()
            center_map = np.copy(instances_map)
            for pt in center_points:
                x, y = int(round(pt[0])), int(round(pt[1]))
                if 0 <= x < w and 0 <= y < h:
                    center_map[y, x] = LaneDetector.CENTER_POINT_MASK
            return center_map
        else:
            return center_points

    def _apply_center_mask(self, instances_map, center_point) -> np.ndarray:
        instances_map = instances_map.cpu().numpy()
        instances_map[int(center_point[0]), int(center_point[1])] = LaneDetector.CENTER_POINT_MASK

        return instances_map

    def _get_segmented_lanes(self, instances_map) -> Optional[Lanes]:
        filtered_instances = self._filter_instances(instances_map)

        # If there are no lanes or only one lane, return None
        # unless there are cached lanes
        if filtered_instances is None or len(filtered_instances) < 2:
            return self._memory.get_lanes()

        center_of_image = instances_map.shape[1] / 2

        lane_left_distances = []
        lane_right_distances = []
        for lane_mask in filtered_instances:
            # Get the lane position as the mean of the x axis
            lane_position = np.mean(np.where(lane_mask > 0)[1]) if np.any(lane_mask) else None

            if lane_position is not None:
                distance_from_center = center_of_image - lane_position

                if distance_from_center < 0: # Right lane, since the distance is negative, and the coords grow from left to right
                    lane_right_distances.append((lane_mask, distance_from_center))
                else: # Left lane, since the distance is positive, and the coords grow from left to right
                    lane_left_distances.append((lane_mask, distance_from_center))

        if len(lane_left_distances) == 0 or len(lane_right_distances) == 0:
            return self._memory.get_lanes()

        # Sort the lanes by their distance from the center, getting the closest lane
        sorted_left_lanes = sorted(lane_left_distances, key=lambda x: x[1])
        sorted_right_lanes = sorted(lane_right_distances, key=lambda x: x[1], reverse=True)

        # Get the masks of the closest lanes
        left_mask, left_lane_distance = sorted_left_lanes[0]
        right_mask, right_lane_distance = sorted_right_lanes[0]

        # Check if the distance between the lanes is too big
        max_distance_threshold = DecoratorConfig().max_distance
        distance = left_lane_distance + abs(right_lane_distance)
        if distance > max_distance_threshold:
            Logger.info(f"Distance between lanes is too big: {distance}")
            return self._memory.get_lanes()

        left_mask, right_mask = self._crop_to_bev(left_mask, right_mask)
        if self._not_enough_points_in_bev_region(left_mask, right_mask):
            return self._memory.get_lanes()

        lanes = Lanes(left_mask, right_mask)
        self._memory.update(lanes)

        return lanes

    def _filter_instances(self, instances_map) -> Optional[List[np.ndarray]]:
        # Get how many unique lanes are there
        unique_instances, inverse_indices = torch.unique(instances_map.view(-1), return_inverse=True)

        # If there is only one lane, return None
        if len(unique_instances) == 1:
            return None

        # For each lane, filter the instances map unique to that lane
        lane_unique_instances = []
        for instance in unique_instances:
            if instance != 0:
                filtered_values = instances_map.clone()
                mask = instances_map != instance
                filtered_values[mask] = 0
                lane_unique_instances.append(filtered_values.cpu().numpy())

        return lane_unique_instances

    def _crop_to_bev(self, *masks) -> tuple[np.ndarray, ...]:
        cropped_masks = []
        for mask in list(masks):
            aux_mask = np.zeros_like(mask)

            min_y = ImageConfig().bev['min_y']
            max_y = ImageConfig().bev['max_y']
            min_x = ImageConfig().bev['min_x']
            max_x = ImageConfig().bev['max_x']
            aux_mask[min_y:max_y, min_x:max_x] = 1

            cropped_mask = mask * aux_mask
            cropped_masks.append(cropped_mask)

        return tuple(cropped_masks)

    def _not_enough_points_in_bev_region(self, *masks) -> bool:
        for mask in list(masks):
            if np.count_nonzero(mask) == 0:
                return True

        return False

    def fit_lanes(self, instance_map, min_points=5, degree=3):
        inst_pred = torch.from_numpy(instance_map) if not torch.is_tensor(instance_map) else instance_map
        assert inst_pred.dim() == 2

        h, w = inst_pred.shape
        inst_pred_expand = inst_pred.view(-1)
        inst_unique = torch.unique(inst_pred_expand)

        # Extract points coordinates for each lane
        lanes = []
        for inst_idx in inst_unique:
            if inst_idx != 0:
                lanes.append(torch.nonzero(inst_pred == inst_idx).cpu().numpy())

        curves = []
        for lane in lanes:
            pts = lane
            # Fitting each lane
            curve = np.polyfit(pts[:, 0], pts[:, 1], 3)
            curves.append(curve)

        return curves

    def sample_from_curve(self, curves, inst_pred, y_sample):
        h, w = inst_pred.shape
        curves_pts = []
        for param in curves:
            # Use new curve function f(y) to calculate x values
            fy = np.poly1d(param)
            x_sample = fy(y_sample)

            # Filter out points beyond image boundaries
            index = np.where(np.logical_or(x_sample < 0, x_sample >= w))
            x_sample[index] = -2

            # Filter out points beyond predictions
            # May filter out bad points, but can also drop good points at the edge
            index = np.where((inst_pred[y_sample, x_sample] == 0).cpu().numpy())
            x_sample[index] = -2

            xy_sample = np.vstack((x_sample, y_sample)).transpose((1, 0)).astype(np.int32)
            curves_pts.append(xy_sample)

        return curves_pts
