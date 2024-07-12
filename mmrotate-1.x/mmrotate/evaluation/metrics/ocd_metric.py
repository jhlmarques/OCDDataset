# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
from collections import OrderedDict

import numpy as np
from PIL import Image
from mmengine.logging import MMLogger
from mmcv.ops import box_iou_quadri, box_iou_rotated
import torch

from mmrotate.evaluation import eval_rbbox_map
from mmrotate.registry import METRICS

from mmrotate.evaluation.metrics.dota_metric import DOTAMetric


@METRICS.register_module()
class OCDMetric(DOTAMetric):

    def __init__(self,
                 path_test_images,
                 *args,
                 **kwargs) -> None:
        
        print(path_test_images)
        assert osp.exists(path_test_images)

        self.path_test_images = path_test_images
        super().__init__(*args, **kwargs)

    def compute_metrics(self, results: list) -> dict:
        pred_bboxes = []
        gt_bboxes = []
        images = []

        for gt, pred in results:
            det = pred['pred_bbox_scores']
            # Change all detections to a base class
            all_dets = np.vstack([det[0], det[1]])
            
            pred['pred_bbox_scores'] = [all_dets]
            gt['labels'] = np.zeros_like(gt['labels'])
            
            pred_bboxes.append(all_dets)
            gt_bboxes.append(gt['bboxes'])

            image = Image.open(osp.join(self.path_test_images, pred['img_id']) + '.png')
            images.append(image)
        
        # Calculate AP over class agnostic predictions
        eval_results_ap = super().compute_metrics(results)

        # Calculate cell count, confluence and polarity errors
        eval_results_bio = self.compute_biological_metrics(pred_bboxes, gt_bboxes, images)

        eval_results = eval_results_ap
        eval_results.update(eval_results_bio)

        return eval_results

    def compute_biological_metrics(self, pred_bboxes, gt_bboxes, images):
        cell_count_errors = []
        confluence_errors = []
        polarity_errors = []
        
        for pred, gt, image in zip(pred_bboxes, gt_bboxes, images):
            
            if len(gt) == 0:
                continue

            pred = pred[pred[:, -1] >= 0.5]            
            pred = pred[:, :-1]
            

            assert pred.shape[-1] == gt.shape[-1]


            # Get metrics from GT
            gt_cell_count = self.calculate_cell_count(gt)
            gt_confluence = self.calculate_confluence(gt, image)
            gt_polarities = self.calculate_polarities(gt)

            # Get metrics from predictions

            pred_cell_count = self.calculate_cell_count(pred)
            pred_confluence = self.calculate_confluence(pred, image)
            pred_polarities = self.calculate_polarities(pred)

            # Compute errors

            cell_count_error = self.calculate_cell_count_error(pred_cell_count, gt_cell_count)
            confluence_error = self.calculate_confluence_error(pred_confluence, gt_confluence)
            polarity_error = self.calculate_polarity_error(pred_polarities, gt_polarities)

            cell_count_errors.append(cell_count_error)
            confluence_errors.append(confluence_error)
            polarity_errors.append(polarity_error)

        eval_results = OrderedDict()
        eval_results['mCount'] = np.mean(cell_count_errors)
        eval_results['mConfl'] = np.mean(confluence_errors)
        eval_results['mPol'] = np.mean(polarity_errors)
        
        return eval_results
    
    def calculate_cell_count(self, bboxes):
        return len(bboxes)

    def calculate_confluence(self, bboxes, image):
        widths = bboxes[:, 2]
        heights = bboxes[:, 3]
        im_w, im_h = image.size
        im_area = im_w * im_h
        box_areas = widths * heights
        sum_of_areas = box_areas.sum()
        
        return sum_of_areas / im_area

    def calculate_polarities(self, bboxes):
        widths = bboxes[:, 2]
        heights = bboxes[:, 3]
        majors = np.maximum(widths, heights)
        minors = np.minimum(widths, heights)

        return majors / minors

    def calculate_cell_count_error(self, pred_cell_count, gt_cell_count):
        return abs(pred_cell_count - gt_cell_count) / gt_cell_count

    def calculate_confluence_error(self, pred_confluence, gt_confluence):
        return abs(pred_confluence - gt_confluence) / gt_confluence

    def calculate_polarity_error(self, pred_polarities, gt_polarities):
        all_polarities = []
        all_polarities.extend(gt_polarities)
        all_polarities.extend(pred_polarities)
        polarities_min = 1.0
        polarities_max = max(all_polarities)
        step = 0.5
        bins_range = np.arange(start=polarities_min, stop=polarities_max, step=step)
        bins_list = [f for f in bins_range]
        hist_gt,  _ = np.histogram(gt_polarities, bins=bins_list)
        hist_model,  _ = np.histogram(pred_polarities, bins=bins_list)

        polarity_chi_distance = 0.0
        for x1, x2 in zip(hist_gt, hist_model):
            denominator = x1
            if denominator == 0:
                continue
            polarity_chi_distance += ((x1-x2)**2) / denominator

        return polarity_chi_distance