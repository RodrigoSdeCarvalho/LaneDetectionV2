import numpy as np
from sklearn.linear_model import LinearRegression
import ujson as json


class LaneEval(object):
    lr = LinearRegression()
    pixel_thresh = 20
    pt_thresh = 0.85
    lane_detection_threshold = 0.85

    @staticmethod
    def get_angle(xs, y_samples):
        xs, ys = xs[xs >= 0], y_samples[xs >= 0]
        if len(xs) > 1:
            LaneEval.lr.fit(ys[:, None], xs)
            k = LaneEval.lr.coef_[0]
            theta = np.arctan(k)
        else:
            theta = 0
        return theta

    @staticmethod
    def line_accuracy(pred, gt, thresh):
        pred = np.array([p if p >= 0 else -100 for p in pred])
        gt = np.array([g if g >= 0 else -100 for g in gt])
        return np.sum(np.where(np.abs(pred - gt) < thresh, 1., 0.)) / len(gt)

    @staticmethod
    def calculate_iou(pred_x_coords, gt_x_coords, y_h_coords, base_thresh):
        """
        Calculate IoU for a single pair of predicted and ground truth lanes.
        pred_x_coords, gt_x_coords: lists/arrays of x-coordinates.
        y_h_coords: list of y-sample heights (for context, length defines iterations).
        base_thresh: The angle-adjusted pixel threshold for this lane pair.
        """
        pred_x_coords = np.array(pred_x_coords)
        gt_x_coords = np.array(gt_x_coords)

        intersections = 0
        unions = 0

        for i in range(len(y_h_coords)):
            px = pred_x_coords[i]
            gx = gt_x_coords[i]

            pred_valid = (px != -2)
            gt_valid = (gx != -2)

            if pred_valid and gt_valid:
                if abs(px - gx) < base_thresh:
                    intersections += 1
                unions += 1
            elif pred_valid or gt_valid:
                unions += 1

        if unions == 0:
            return 0.0
        return float(intersections) / unions

    @staticmethod
    def calculate_f1_score(tp, fp, fn):
        """
        Calculate F1-Score given true positives, false positives, and false negatives.
        """
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return f1, precision, recall

    @staticmethod
    def is_lane_detected(pred_lane, gt_lane, thresh):
        """
        Check if a predicted lane matches a ground truth lane based on the detection threshold.
        Returns True if the lane is considered detected (accuracy >= threshold).
        """
        pred = np.array([p if p >= 0 else -100 for p in pred_lane])
        gt = np.array([g if g >= 0 else -100 for g in gt_lane])
        valid_points = np.sum(gt != -100)
        if valid_points == 0:
            return False
        correct_points = np.sum(np.where(np.abs(pred - gt) < thresh, 1., 0.))
        accuracy = correct_points / valid_points
        return accuracy >= LaneEval.lane_detection_threshold

    @staticmethod
    def bench(pred, gt, y_samples, running_time):
        if any(len(p) != len(y_samples) for p in pred):
            raise Exception('Format of lanes error.')
        if running_time > 200 or len(gt) + 2 < len(pred):
            return 0., 0., 1., 0., 0., 0., 0.  
        
        angles = [LaneEval.get_angle(np.array(x_gts), np.array(y_samples)) for x_gts in gt]
        threshs = [LaneEval.pixel_thresh / np.cos(angle) for angle in angles]
        
        line_accs = []
        fp, fn = 0., 0.
        matched = 0.
        ious = []  
        
        for x_gts, thresh in zip(gt, threshs):
            accs = [LaneEval.line_accuracy(np.array(x_preds), np.array(x_gts), thresh) for x_preds in pred]
            max_acc = np.max(accs) if len(accs) > 0 else 0.
            
            if max_acc < LaneEval.pt_thresh:
                fn += 1
            else:
                matched += 1
                best_pred_idx = np.argmax(accs)
                iou = LaneEval.calculate_iou(pred[best_pred_idx], x_gts, y_samples, thresh)
                ious.append(iou)
            line_accs.append(max_acc)
            
        fp = len(pred) - matched
        if len(gt) > 4 and fn > 0:
            fn -= 1
        s = sum(line_accs)
        if len(gt) > 4:
            s -= min(line_accs)
        avg_iou = np.mean(ious) if ious else 0.0  
        
        tp = 0 
        fp_f1 = 0  
        fn_f1 = 0  
        
        gt_matched = [False] * len(gt)
        
        for pred_lane in pred:
            best_match_idx = -1
            best_match_acc = 0
            
            for i, (gt_lane, thresh) in enumerate(zip(gt, threshs)):
                if not gt_matched[i]:  
                    if LaneEval.is_lane_detected(pred_lane, gt_lane, thresh):
                        acc = LaneEval.line_accuracy(np.array(pred_lane), np.array(gt_lane), thresh)
                        if acc > best_match_acc:
                            best_match_acc = acc
                            best_match_idx = i
            
            if best_match_idx != -1:
                tp += 1
                gt_matched[best_match_idx] = True
            else:
                fp_f1 += 1
        
        fn_f1 = sum(1 for matched in gt_matched if not matched)
        
        f1_score, precision, recall = LaneEval.calculate_f1_score(tp, fp_f1, fn_f1)
        
        return s / max(min(4.0, len(gt)), 1.), fp / len(pred) if len(pred) > 0 else 0., fn / max(min(len(gt), 4.) , 1.), avg_iou, f1_score, precision, recall

    @staticmethod
    def bench_one_submit(pred_file, gt_file):
        try:
            json_pred = [json.loads(line) for line in open(pred_file).readlines()]
        except BaseException as e:
            raise Exception('Fail to load json file of the prediction.')
        json_gt = [json.loads(line) for line in open(gt_file).readlines()]
        if len(json_gt) != len(json_pred):
            raise Exception('We do not get the predictions of all the test tasks')
        gts = {l['raw_file']: l for l in json_gt}
        accuracy, fp, fn, iou, f1_score, precision, recall = 0., 0., 0., 0., 0., 0., 0.
        for pred in json_pred:
            if 'raw_file' not in pred or 'lanes' not in pred or 'run_time' not in pred:
                raise Exception('raw_file or lanes or run_time not in some predictions.')
            raw_file = pred['raw_file']
            pred_lanes = pred['lanes']
            run_time = pred['run_time']
            if raw_file not in gts:
                raise Exception('Some raw_file from your predictions do not exist in the test tasks.')
            gt = gts[raw_file]
            gt_lanes = gt['lanes']
            y_samples = gt['h_samples']
            try:
                a, p, n, i, f1, prec, rec = LaneEval.bench(pred_lanes, gt_lanes, y_samples, run_time)
            except BaseException as e:
                raise Exception('Format of lanes error.')
            accuracy += a
            fp += p
            fn += n
            iou += i
            f1_score += f1
            precision += prec
            recall += rec
        num = len(gts)
        return json.dumps([
            {'name': 'Accuracy', 'value': accuracy / num, 'order': 'desc'},
            {'name': 'FP', 'value': fp / num, 'order': 'asc'},
            {'name': 'FN', 'value': fn / num, 'order': 'asc'},
            {'name': 'IoU', 'value': iou / num, 'order': 'desc'},
            {'name': 'F1-Score', 'value': f1_score / num, 'order': 'desc'},
            {'name': 'Precision', 'value': precision / num, 'order': 'desc'},
            {'name': 'Recall', 'value': recall / num, 'order': 'desc'}
        ])


if __name__ == '__main__':
    import sys
    try:
        if len(sys.argv) != 3:
            raise Exception('Invalid input arguments. Usage: python tusimple_test_metrics.py <prediction_file> <ground_truth_file>')
        print(LaneEval.bench_one_submit(sys.argv[1], sys.argv[2]))
    except Exception as e:
        print(str(e))
        sys.exit(1)
