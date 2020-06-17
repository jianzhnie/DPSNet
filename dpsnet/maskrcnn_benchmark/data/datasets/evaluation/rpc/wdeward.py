import json
import logging
import os
from datetime import datetime
import numpy as np
import mmcv
import boxx
import rpctool
from tqdm import tqdm

from maskrcnn_benchmark.utils.density import rpc_category_to_super_category

LEVELS = ('easy', 'medium', 'hard', 'averaged')
NUM_CLASSES = 200 + 1  # 1-based
THRESHOLD = 0.95


def get_cAcc(result, level):
    index = LEVELS.index(level)
    return float(result.loc[index, 'cAcc'].strip('%'))


def check_best_result(output_folder, result, result_str, filename):
    current_cAcc = get_cAcc(result, 'averaged')
    best_path = os.path.join(output_folder, 'best_result.txt')
    if os.path.exists(best_path):
        with open(best_path) as f:
            best_cAcc = float(f.readline().strip())
        if current_cAcc >= best_cAcc:
            best_cAcc = current_cAcc
            with open(best_path, 'w') as f:
                f.write(str(best_cAcc) + '\n' + filename + '\n' + result_str)
    else:
        best_cAcc = current_cAcc
        with open(best_path, 'w') as f:
            f.write(str(current_cAcc) + '\n' + filename + '\n' + result_str)
    return best_cAcc


def rpc_evaluation(dataset,
                   predictions,
                   output_folder,
                   generate_pseudo_labels=False,
                   iteration=-1,
                   threshold=THRESHOLD,
                   use_ground_truth=False,  # use ground truth to select pseudo labels
                   **_):
    threshold = 0.9995 if threshold >= 1 else threshold

    logger = logging.getLogger("maskrcnn_benchmark.inference")
    if generate_pseudo_labels:
        logger.info('Use ground truth: {}'.format(use_ground_truth))

    pred_boxlists = []
    pred_boxlists_with_density = []
    annotations = []
    density_correct = 0
    box_correct = 0
    mae = 0  # MEAN ABSOLUTE ERROR
    has_density_map = predictions[0].has_field('density')
    num_density_classes = 1
    if has_density_map:
        num_density_classes = predictions[0].get_field('density').shape[-1]
        logger.info('Density category: {}'.format(num_density_classes))

    for image_id, prediction in tqdm(enumerate(predictions)):
        img_info = dataset.get_img_info(image_id)

        image_width = img_info["width"]
        image_height = img_info["height"]
        prediction = prediction.resize((image_width, image_height))
        bboxes = prediction.bbox.tolist()
        labels = prediction.get_field("labels").tolist()
        scores = prediction.get_field("scores").tolist()

        # -----------------------------------------------#
        # -----------------Pseudo Label------------------#
        # -----------------------------------------------#

        gt_density_cat_counts = np.zeros((num_density_classes,), dtype=np.int32)
        gt_all_cat_counts = np.zeros((NUM_CLASSES,), dtype=np.int32)
        pred_density_cat_counts = np.zeros((num_density_classes,), dtype=np.int32)
        # density = 0.0
        if has_density_map:
            pred_density_cat_counts = prediction.get_field('density').numpy()
            pred_density_cat_map = prediction.get_field('density_map').numpy()
            pred_density_cat_counts = np.round(pred_density_cat_counts).astype(np.int32)

            ann = dataset.get_annotation(img_info['id'])
            for category, x, y, w, h in ann:
                density_category = rpc_category_to_super_category(category, num_classes=num_density_classes)
                gt_all_cat_counts[category] += 1
                gt_density_cat_counts[density_category] += 1

            is_correct = np.all(gt_density_cat_counts == pred_density_cat_counts)
            if is_correct:
                density_correct += 1
            else:
                mae += np.sum(np.abs(gt_density_cat_counts - pred_density_cat_counts))

        box_density_cat_counts = np.zeros((num_density_classes,), dtype=np.int32)
        box_all_cat_counts = np.zeros((NUM_CLASSES,), dtype=np.int32)

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
                    box = bboxes[i]
                    label = labels[i]
                    density_category = rpc_category_to_super_category(label, num_density_classes)
                    box_all_cat_counts[label] += 1
                    box_density_cat_counts[density_category] += 1
                    x, y, width, height = box[0], box[1], box[2] - box[0], box[3] - box[1]
                    image_result['bbox'].append(
                        (label, x, y, width, height)
                    )
            if use_ground_truth:
                is_valid = np.all(box_all_cat_counts == gt_all_cat_counts)
            else:
                is_valid = np.all(box_density_cat_counts == pred_density_cat_counts)
            if is_valid:
                annotations.append(image_result)
                is_box_correct = np.all(box_all_cat_counts == gt_all_cat_counts)
                if is_box_correct:
                    box_correct += 1

        # -----------------------------------------------#
        # -----------------------------------------------#
        # -----------------------------------------------#

        for i in range(len(prediction)):
            score = scores[i]
            box = bboxes[i]
            label = labels[i]

            x, y, width, height = box[0], box[1], box[2] - box[0], box[3] - box[1]

            pred_boxlists.append({
                "image_id": img_info['id'],
                "category_id": int(label),
                "bbox": [float(k) for k in [x, y, width, height]],
                "score": float(score),
            })
            
            pred_boxlists_with_density.append({
                "image_id": img_info['id'],
                "category_id": int(label),
                "bbox": [float(k) for k in [x, y, width, height]],
                "score": float(score),
                "density_map":pred_density_cat_map
            })

    if has_density_map:
        logger.info('Density Ratio: {:.3f}'.format(density_correct / len(predictions)))
        logger.info('Density MAE  : {:.3f} '.format(mae / len(predictions)))
        if generate_pseudo_labels:
            if len(annotations) == 0:
                logger.info('No annotations are selected.')
            else:
                logger.info(
                    'Select  Ratio: {:.3f} ({}/{}, {:.5f} Threshold)'.format(box_correct / len(annotations),
                                                                             box_correct,
                                                                             len(annotations),
                                                                             threshold))

    time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if len(pred_boxlists) == 0:
        logger.info('Nothing detected.')
        with open(os.path.join(output_folder, 'result_{}.txt'.format(time_stamp)), 'w') as fid:
            fid.write('Nothing detected.')
        return dict(metrics={})

    if generate_pseudo_labels:
        logger.info('Pseudo-Labeling: {}'.format(len(annotations)))
        with open(os.path.join(output_folder, 'pseudo_labeling.json'), 'w') as fid:
            json.dump(annotations, fid)

    save_path = os.path.join(output_folder, 'bbox_results.json')
    with open(save_path, 'w') as fid:
        json.dump(pred_boxlists, fid)
        
    mmcv.dump(pred_boxlists_with_density, os.path.join(output_folder, 'bbox_results_with_density.pkl'))
    
    res_js = boxx.loadjson(save_path)
    ann_js = boxx.loadjson(dataset.ann_file)
    result = rpctool.evaluate(res_js, ann_js)
    logger.info(result)

    result_str = str(result)
    if iteration > 0:
        filename = os.path.join(output_folder, 'result_{:07d}.txt'.format(iteration))
    else:
        filename = os.path.join(output_folder, 'result_{}.txt'.format(time_stamp))

    if has_density_map:
        result_str += '\n' + 'Ratio: {:.3f}, '.format(density_correct / len(predictions)) + 'MAE: {:.3f} '.format(mae / len(predictions))
    with open(filename, 'w') as fid:
        fid.write(result_str)

    best_cAcc = check_best_result(output_folder, result, result_str, filename)
    logger.info('Best cAcc: {}%'.format(best_cAcc))
    metrics = {
        'cAcc': {
            'averaged': get_cAcc(result, 'averaged'),
            'hard': get_cAcc(result, 'hard'),
            'medium': get_cAcc(result, 'medium'),
            'easy': get_cAcc(result, 'easy'),
        }
    }
    if has_density_map:
        metrics.update({
            'Ratio': density_correct / len(predictions),
            'MAE': mae / len(predictions),
        })
    eval_result = dict(metrics=metrics)
    return eval_result
