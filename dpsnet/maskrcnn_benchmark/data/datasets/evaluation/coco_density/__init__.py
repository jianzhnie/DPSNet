import json
import logging
import os
from datetime import datetime

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from maskrcnn_benchmark.utils.density import contiguous_coco_category_to_super_category


def coco_density_evaluation(dataset, predictions, output_folder, iteration=-1, generate_pseudo_labels=True, has_annotation=True, **_):
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    use_ground_truth = False
    threshold = 0.95
    coco_results = []
    annotations = []
    density_correct = 0
    box_correct = 0
    mae = 0  # MEAN ABSOLUTE ERROR
    metrics = {}
    num_density_classes = 1
    has_density_map = predictions[0].has_field('density')
    if has_density_map:
        num_density_classes = predictions[0].get_field('density').shape[-1]
        logger.info('Density category: {}'.format(num_density_classes))
    for image_id, prediction in enumerate(predictions):
        if len(prediction) == 0:
            continue

        img_info = dataset.get_img_info(image_id)
        original_id = img_info['id']
        image_width = img_info["width"]
        image_height = img_info["height"]
        prediction = prediction.resize((image_width, image_height))
        prediction = prediction.convert("xywh")

        boxes = prediction.bbox.tolist()
        scores = prediction.get_field("scores").tolist()
        labels = prediction.get_field("labels").tolist()

        mapped_labels = [dataset.contiguous_category_id_to_json_id[i] for i in labels]

        gt_super_cat_counts = np.zeros((num_density_classes,), dtype=np.int32)
        gt_all_cat_counts = np.zeros((81,), dtype=np.int32)
        pred_super_cat_counts = np.zeros((num_density_classes,), dtype=np.int32)
        if has_density_map:
            pred_super_cat_counts = prediction.get_field('density').numpy()
            pred_super_cat_counts = np.round(pred_super_cat_counts).astype(np.int32)
            if has_annotation:
                ann = dataset.get_annotation(img_info['id'])
                for category in ann['labels']:
                    super_category = contiguous_coco_category_to_super_category(category, num_classes=num_density_classes)
                    gt_all_cat_counts[category] += 1
                    gt_super_cat_counts[super_category] += 1

                is_correct = np.all(gt_super_cat_counts == pred_super_cat_counts)
                if is_correct:
                    density_correct += 1
                else:
                    mae += np.sum(np.abs(gt_super_cat_counts - pred_super_cat_counts))

        box_super_cat_counts = np.zeros((num_density_classes,), dtype=np.int32)
        box_all_cat_counts = np.zeros((81,), dtype=np.int32)
        if generate_pseudo_labels and has_density_map:
            image_result = {
                'bbox': [],
                'width': image_width,
                'height': image_height,
                'id': img_info['id'],
                'file_name': img_info['file_name'],
            }

            for i in range(len(prediction)):
                score = scores[i]
                if score > threshold:
                    box = boxes[i]
                    label = labels[i]
                    super_category = contiguous_coco_category_to_super_category(label, num_classes=num_density_classes)
                    box_all_cat_counts[label] += 1
                    box_super_cat_counts[super_category] += 1
                    x, y, width, height = box
                    image_result['bbox'].append(
                        (label, x, y, width, height)
                    )
            if use_ground_truth and has_annotation:
                is_valid = np.all(box_all_cat_counts == gt_all_cat_counts)
            else:
                is_valid = np.all(box_super_cat_counts == pred_super_cat_counts)
            if is_valid:
                annotations.append(image_result)
                if has_annotation:
                    is_box_correct = np.all(box_all_cat_counts == gt_all_cat_counts)
                    if is_box_correct:
                        box_correct += 1

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": mapped_labels[k],
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )

    if has_density_map:
        metrics['ratio'] = density_correct / len(predictions)
        metrics['mae'] = mae / len(predictions)
        logger.info('Density Ratio: {:.3f}'.format(density_correct / len(predictions)))
        logger.info('Density MAE  : {:.3f} '.format(mae / len(predictions)))
        if generate_pseudo_labels:
            if len(annotations) == 0:
                logger.info('No annotations are selected.')
            else:
                metrics['select_ratio'] = box_correct / len(annotations)
                metrics['pseudo_labels'] = len(annotations)
                logger.info(
                    'Select  Ratio: {:.3f} ({}/{}, {:.5f} Threshold)'.format(box_correct / len(annotations),
                                                                             box_correct,
                                                                             len(annotations),
                                                                             threshold))

    time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if len(coco_results) == 0:
        logger.info('Nothing detected.')
        with open(os.path.join(output_folder, 'result_{}.txt'.format(time_stamp)), 'w') as fid:
            fid.write('Nothing detected.')
        return dict(metrics={})

    if generate_pseudo_labels:
        logger.info('Pseudo-Labeling: {}'.format(len(annotations)))
        with open(os.path.join(output_folder, 'pseudo_labeling.json'), 'w') as fid:
            json.dump(annotations, fid)

    if not has_annotation:
        return dict(metrics=metrics)

    file_path = os.path.join(output_folder, "bbox.json")

    with open(file_path, "w") as f:
        json.dump(coco_results, f)

    coco_gt = dataset.coco
    coco_dt = coco_gt.loadRes(str(file_path)) if coco_results else COCO()

    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    result_strings = []
    keys = ["AP", "AP50", "AP75", "APs", "APm", "APl"]

    for i, key in enumerate(keys):
        metrics[key] = coco_eval.stats[i]
        logger.info('{:<10}: {}'.format(key, round(coco_eval.stats[i], 3)))
        result_strings.append('{:<10}: {}'.format(key, round(coco_eval.stats[i], 3)))

    if iteration > 0:
        filename = os.path.join(output_folder, 'result_{:07d}.txt'.format(iteration))
    else:
        filename = os.path.join(output_folder, 'result_{}.txt'.format(time_stamp))

    with open(filename, "w") as f:
        f.write('\n'.join(result_strings))

    return dict(metrics=metrics)
