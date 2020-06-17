# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import collections
import datetime
import logging
import os
import time
from bisect import bisect_right
import torch
import torch.distributed as dist
from torch.utils.data import Subset
import numpy as np
from maskrcnn_benchmark.data.build import build_dataset, make_data_loader
from maskrcnn_benchmark.data.datasets import RPCPseudoDataset, ConcatDataset, RPCTestDataset
from maskrcnn_benchmark.data.transforms import build_transforms
from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.utils.comm import get_rank
from tools.test_net import do_test


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def write_metric(eval_result, prefix, summary_writer, global_step):
    for key in eval_result:
        value = eval_result[key]
        tag = '{}/{}'.format(prefix, key)
        if isinstance(value, collections.Mapping):
            write_metric(value, tag, summary_writer, global_step)
        else:
            summary_writer.add_scalar(tag, value, global_step=global_step)


def do_train(
        cfg,
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
        distributed
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()

    summary_writer = None
    if get_rank() == 0:
        import tensorboardX
        summary_writer = tensorboardX.SummaryWriter(os.path.join(checkpointer.save_dir, 'tf_logs'))

    threshold = 0.95
    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        scheduler.step()

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            if summary_writer:
                summary_writer.add_scalar('loss/total_loss', losses_reduced, global_step=iteration)
                for name, value in loss_dict_reduced.items():
                    summary_writer.add_scalar('loss/%s' % name, value, global_step=iteration)
                summary_writer.add_scalar('lr', optimizer.param_groups[0]["lr"], global_step=iteration)

            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
            if iteration != max_iter:
                eval_results = do_test(cfg, model, distributed, threshold, iteration=iteration)
                if get_rank() == 0 and summary_writer:  # only on main thread results are returned.
                    for eval_result, dataset in zip(eval_results, cfg.DATASETS.TEST):
                        write_metric(eval_result['metrics'], 'metrics/' + dataset, summary_writer, iteration)
                model.train()  # *IMPORTANT*: restore train state
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )


def cross_do_train(
        cfg,
        model,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
        distributed
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start cross training!")
    meters = MetricLogger(delimiter="  ")
    max_iter = cfg.SOLVER.MAX_ITER
    start_iter = arguments["iteration"]
    model.train()
    # ----------------prepare----------------
    # ---------------------------------------
    # ---------------------------------------
    is_train = True
    paths_catalog = import_file(
        "maskrcnn_benchmark.config.paths_catalog", cfg.PATHS_CATALOG, True
    )
    DatasetCatalog = paths_catalog.DatasetCatalog
    transforms = build_transforms(cfg, is_train=is_train)
    dataset_list = cfg.DATASETS.TRAIN

    start_training_time = time.time()
    end = time.time()

    summary_writer = None
    if get_rank() == 0:
        import tensorboardX
        summary_writer = tensorboardX.SummaryWriter(os.path.join(checkpointer.save_dir, 'tf_logs'))

    ann_file = cfg.TEST.PSEUDO_LABELS_ANN_FILE
    images_dir = cfg.TEST.TEST_IMAGES_DIR

    iteration = start_iter
    total_steps = cfg.SOLVER.CROSS_TRAIN_STEPS
    for step in range(total_steps):
        logger.info('Start training {}th/{} step'.format(step + 1, total_steps))
        iter_per_step = cfg.SOLVER.ITER_PER_STEP

        pseudo_dataset = RPCPseudoDataset(images_dir=images_dir, ann_file=ann_file, use_density_map=True, transforms=transforms)
        # ---------------------------------------------------------------------------------
        pseudo_dataset.density_categories = cfg.MODEL.DENSITY_HEAD.NUM_CLASSES
        pseudo_dataset.density_map_stride = cfg.MODEL.DENSITY_HEAD.FPN_LEVEL_STRIDE
        min_sigmas = {
            1: 1.0,
            2: 0.5,
            3: 0.333,
        }
        min_sigma = min_sigmas[cfg.MODEL.DENSITY_HEAD.FPN_LEVEL]
        pseudo_dataset.density_min_sigma = min_sigma
        print('using density_min_sigma: {}'.format(min_sigma))
        # ---------------------------------------------------------------------------------

        train_datasets = build_dataset(cfg, dataset_list, transforms, DatasetCatalog, is_train)

        ratio = cfg.SOLVER.CROSS_TRAIN_DATA_RATIO
        if ratio > 0:  # dynamic source dataset according to pseudo dataset
            assert len(train_datasets) == 1
            train_size = len(train_datasets[0])
            indices = np.arange(train_size)
            train_size = min(train_size, int(ratio * len(pseudo_dataset)))
            indices = np.random.choice(indices, size=train_size, replace=False)
            subset_dataset = Subset(train_datasets[0], indices=indices)
            train_datasets = [subset_dataset]
        elif ratio < 0:  # fixed size source dataset
            assert len(train_datasets) == 1
            train_size = len(train_datasets[0])
            indices = np.arange(train_size)
            train_size = min(train_size, abs(ratio))
            indices = np.random.choice(indices, size=train_size, replace=False)
            subset_dataset = Subset(train_datasets[0], indices=indices)
            train_datasets = [subset_dataset]

        datasets_s = train_datasets + [pseudo_dataset]
        datasets_s = ConcatDataset(datasets_s)

        # logger.info('Subset train dataset: {}'.format(len(subset_dataset)))
        logger.info('Pseudo train dataset: {}'.format(len(pseudo_dataset)))
        logger.info('Source train dataset: {}'.format(len(datasets_s)))

        data_loader_t = make_data_loader(
            cfg,
            is_train=is_train,
            is_distributed=distributed,
            start_iter=0,
            datasets=[datasets_s],
            num_iters=iter_per_step
        )

        thresholds = [0.8, 0.85, 0.9, 0.95]
        threshold = thresholds[bisect_right([5, 8, 9], step)]
        for (images_t, targets_t, _) in data_loader_t:
            data_time = time.time() - end
            iteration = iteration + 1
            arguments["iteration"] = iteration

            scheduler.step()

            # images_s = images_s.to(device)
            # targets_s = [target.to(device) for target in targets_s]
            # loss_dict_s = model(images_s, targets_s, is_target_domain=False)
            # loss_dict_s = {key + '_s': value for key, value in loss_dict_s.items()}

            images_t = images_t.to(device)
            targets_t = [target.to(device) for target in targets_t]
            loss_dict = model(images_t, targets_t, is_target_domain=True)

            # loss_dict.update(loss_dict_s)

            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            meters.update(loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)

            eta_seconds = meters.time.global_avg * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if iteration % 20 == 0 or iteration == max_iter:
                if summary_writer:
                    summary_writer.add_scalar('loss/total_loss', losses_reduced, global_step=iteration)
                    for name, value in loss_dict_reduced.items():
                        summary_writer.add_scalar('loss/%s' % name, value, global_step=iteration)
                    summary_writer.add_scalar('lr', optimizer.param_groups[0]["lr"], global_step=iteration)

                logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "iter: {iter}",
                            "{meters}",
                            "lr: {lr:.6f}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        eta=eta_string,
                        iter=iteration,
                        meters=str(meters),
                        lr=optimizer.param_groups[0]["lr"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )
            if iteration % checkpoint_period == 0:
                checkpointer.save("model_{:07d}".format(iteration), **arguments)
                if iteration != max_iter:
                    eval_results = do_test(cfg, model, distributed, threshold, iteration=iteration)
                    if get_rank() == 0 and summary_writer:  # only on main thread results are returned.
                        for eval_result, dataset in zip(eval_results, cfg.DATASETS.TEST):
                            write_metric(eval_result['metrics'], 'metrics/' + dataset, summary_writer, iteration)
                    model.train()  # restore train state

        logger.info('Generating new pseudo labels...')
        test_dataset = RPCTestDataset(images_dir=cfg.TEST.TEST_IMAGES_DIR,
                                      ann_file=cfg.TEST.TEST_ANN_FILE,
                                      transforms=build_transforms(cfg, is_train=False))
        dataset_name = 'rpc_2019_test'
        dataset_names = [dataset_name]
        eval_results = do_test(cfg, model, distributed, threshold, iteration=iteration, generate_pseudo_labels=True, dataset_names=dataset_names,
                               datasets=[test_dataset], use_ground_truth=True)
        if get_rank() == 0 and summary_writer:  # only on main thread results are returned.
            for eval_result, dataset in zip(eval_results, dataset_names):
                write_metric(eval_result['metrics'], 'metrics/' + dataset, summary_writer, iteration)
        model.train()  # restore train state
        ann_file = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name, 'pseudo_labeling.json')

    checkpointer.save("model_final", **arguments)
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
