import os
from argparse import ArgumentParser
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from utils.public_utils import fix_train_random_seed, setup_device, cal_torch_model_params, get_tqdm_desc, is_left_better_right
from dataloader.data_utils import simple_transforms_for_train, aug2_transforms_for_train, transforms_for_eval, load_video_data_list, load_multi_view_split_file, simple_transforms_no_aug
from finetune.train_utils import get_scaffold_regression_metainfo
from dataloader.dataset import FrameDataset
from finetune.train_utils import train_one_epoch, save_finetune_ckpt
from finetune.train_utils import evaluate_on_video
from model.model_utils import load_pretrained_component
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from utils.train_utils import load_smiles
from utils.splitter import *
from model.base.predictor import FramePredictor, RegMeanStdWrapper


def parse_args():
    # Parse arguments
    parser = ArgumentParser(description='PyTorch Implementation of videoMol')

    # basic
    parser.add_argument('--dataroot', type=str, default="../datasets/fine-tuning/", help='data root')
    parser.add_argument('--dataset', type=str, help='dataset name')
    parser.add_argument('--label_column_name', type=str, default="label", help='column name of label')
    parser.add_argument('--video_dir_name', type=str, default="video-224x224", help='directory name of video')
    parser.add_argument('--gpu', type=str, default="0", help='GPUs of CUDA_VISIBLE_DEVICES, e.g. 0,1,2,3')
    parser.add_argument('--ngpu', type=int, default=8, help='number of GPUs to use')
    parser.add_argument('--workers', default=2, type=int, help='number of data loading workers (default: 2)')

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay")

    # train
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42) to split dataset')
    parser.add_argument('--runseed', type=int, default=2021, help='random seed to run model (default: 2021)')
    parser.add_argument('--split', default="random", type=str,
                        choices=['multi_view_split_file', 'split_file', 'random', 'stratified', 'scaffold',
                                 'random_scaffold', 'balanced_scaffold'], help='regularization of classification loss')
    parser.add_argument('--split_path', type=str, help='e.g. ./dataset/mpp/data_name/scaffold.npy')
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs")
    parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts) (default: 0)')
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--resume", type=str, default=None, help='Resume training from a path of checkpoint')
    parser.add_argument('--arch', default="arch1", type=str, choices=['arch1', 'arch2', 'arch3'], help='prediction classifier (fine-tuning stage)')
    parser.add_argument('--imageSize', type=int, default=224, help='the height / width of the input image to network')
    parser.add_argument('--model_name', type=str, default="vit_small_patch16_224", help='model name')
    parser.add_argument('--n_frame', type=int, default=60, help='the number of frame')
    parser.add_argument('--close_image_aug', action='store_true', default=False, help='whether to close data augmentation')
    parser.add_argument('--image_aug_name', type=str, default="simple", choices=["simple", "aug2"], help='whether to close data augmentation')
    parser.add_argument('--task_type', type=str, default="classification", choices=["classification", "regression"], help='task type')
    parser.add_argument('--save_finetune_ckpt', type=int, default=1, choices=[0, 1], help='1 represents saving best ckpt, 0 represents no saving best ckpt')

    # log
    parser.add_argument('--log_dir', default='./logs/finetune_videomol/', help='path to log')

    # Parse arguments
    return parser.parse_args()


def main(args):
    ########################## basic
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    args.csv_file = f"{args.dataroot}/{args.dataset}/processed/{args.dataset}_processed_ac.csv"

    # creating log dir
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # gpus
    device, device_ids = setup_device(args.ngpu)

    # fix random seeds
    fix_train_random_seed(args.runseed)

    ######################### basic information of data
    stat_mean, stat_std, inverse_scale_mean_std = None, None, None
    if args.task_type == "classification":
        eval_metric = "rocauc"
        valid_select = "max"
        min_value = -np.inf
        criterion = nn.BCEWithLogitsLoss(reduction="none")
    elif args.task_type == "regression":
        if args.dataset == "qm7" or args.dataset == "qm8" or args.dataset == "qm9":
            eval_metric = "mae"
            criterion = nn.L1Loss()
        else:
            eval_metric = "rmse"
            criterion = nn.MSELoss()
        valid_select = "min"
        min_value = np.inf

        task_metainfo = get_scaffold_regression_metainfo()
        if args.dataset in task_metainfo:
            # for regression task, pre-compute mean and std
            stat_mean = task_metainfo[args.dataset]["mean"]
            stat_std = task_metainfo[args.dataset]["std"]
            inverse_scale_mean_std = (stat_mean, stat_std)
    else:
        raise Exception("param {} is not supported".format(args.task_type))

    print("eval_metric: {}".format(eval_metric))

    ########################## load dataset
    # transforms
    if args.close_image_aug:
        train_transforms = simple_transforms_no_aug(mean_std=(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
        eval_transforms = simple_transforms_no_aug(mean_std=(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    else:
        if args.image_aug_name == "simple":
            train_transforms = simple_transforms_for_train(resize=args.imageSize, mean_std=(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
        elif args.image_aug_name == "aug2":
            train_transforms = aug2_transforms_for_train(resize=args.imageSize, mean_std=(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))

        eval_transforms = transforms_for_eval(resize=(args.imageSize, args.imageSize),
                                              img_size=(args.imageSize, args.imageSize),
                                              mean_std=(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))

    video_index_list, video_path_list, video_label_list = load_video_data_list(args.dataroot, args.dataset, label_column_name=args.label_column_name, video_dir_name=args.video_dir_name, csv_suffix="_processed_ac")
    video_index_list, video_path_list, video_labels = np.array(video_index_list), np.array(video_path_list), np.array(video_label_list).reshape(len(video_label_list), -1)
    if args.task_type == "classification":
        video_labels = video_labels.astype(int)
    elif args.task_type == "regression":
        video_labels = video_labels.astype(float)
    num_videos, num_tasks = video_labels.shape

    # split train/valid/test
    train_video_path_list, valid_video_path_list, test_video_path_list = None, None, None
    train_video_labels, valid_video_labels, test_video_labels = None, None, None
    train_video_index_list, valid_video_index_list, test_video_index_list = None, None, None
    if args.split == "multi_view_split_file":
        video_root = f"{args.dataroot}/{args.dataset}/processed/{args.video_dir_name}"
        train_idx, val_idx, test_idx, \
            train_video_index_list, valid_video_index_list, test_video_index_list, \
            train_video_path_list, valid_video_path_list, test_video_path_list, \
            train_video_labels, valid_video_labels, test_video_labels = \
            load_multi_view_split_file(video_root, args.split_path)
    elif args.split == "split_file":
        train_idx, val_idx, test_idx = split_train_val_test_idx_split_file(sort=False, split_path=args.split_path)
    elif args.split == "random":
        train_idx, val_idx, test_idx = split_train_val_test_idx(list(range(0, num_videos)), frac_train=0.8,
                                                                frac_valid=0.1, frac_test=0.1, seed=args.seed)
    elif args.split == "stratified":
        train_idx, val_idx, test_idx = split_train_val_test_idx_stratified(list(range(0, num_videos)), video_labels,
                                                                           frac_train=0.8, frac_valid=0.1,
                                                                           frac_test=0.1, seed=args.seed)
    elif args.split == "scaffold":
        smiles = load_smiles(args.csv_file)
        train_idx, val_idx, test_idx = scaffold_split_train_val_test(list(range(0, num_videos)), smiles, frac_train=0.8,
                                                                     frac_valid=0.1, frac_test=0.1)
    elif args.split == "random_scaffold":
        smiles = load_smiles(args.csv_file)
        train_idx, val_idx, test_idx = random_scaffold_split_train_val_test(list(range(0, num_videos)), smiles,
                                                                            frac_train=0.8, frac_valid=0.1,
                                                                            frac_test=0.1, seed=args.seed)

    elif args.split == "balanced_scaffold":
        smiles = load_smiles(args.csv_file)
        train_idx, val_idx, test_idx = scaffold_split_balanced_train_val_test(list(range(0, num_videos)), smiles,
                                                                              frac_train=0.8, frac_valid=0.1,
                                                                              frac_test=0.1, seed=args.seed,
                                                                              balanced=True)
    else:
        raise Exception(f"{args.split} is not support.")

    if args.split != "multi_view_split_file":
        train_video_index_list, train_video_path_list, train_video_labels = video_index_list[train_idx], video_path_list[train_idx], video_labels[train_idx]
        valid_video_index_list, valid_video_path_list, valid_video_labels = video_index_list[val_idx], video_path_list[val_idx], video_labels[val_idx]
        test_video_index_list, test_video_path_list, test_video_labels = video_index_list[test_idx], video_path_list[test_idx], video_labels[test_idx]

    if stat_mean is not None and stat_std is not None:
        print("transforming train data ...")
        train_video_labels = (train_video_labels - stat_mean) / stat_std

    train_frameDataset = FrameDataset(train_video_path_list, video_labels=train_video_labels, video_indexs=train_video_index_list, transforms=train_transforms, n_frame=args.n_frame, ret_index=True)
    valid_frameDataset = FrameDataset(valid_video_path_list, video_labels=valid_video_labels, video_indexs=valid_video_index_list, transforms=eval_transforms, n_frame=args.n_frame, ret_index=True)
    test_frameDataset = FrameDataset(test_video_path_list, video_labels=test_video_labels, video_indexs=test_video_index_list, transforms=eval_transforms, n_frame=args.n_frame, ret_index=True)

    train_dataloader = DataLoader(train_frameDataset, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True)
    valid_dataloader = DataLoader(valid_frameDataset, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)
    test_dataloader = DataLoader(test_frameDataset, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)

    ########################## load model
    # Load model
    print("add mlp [{}] for downstream task".format(args.arch))
    videoMol = FramePredictor(model_name=args.model_name, head_arch=args.arch, num_tasks=num_tasks, pretrained=False)
    if stat_mean and stat_std:
        videoMol = RegMeanStdWrapper(model=videoMol, stat_mean=stat_mean, stat_std=stat_std, evaluate=False,
                                     device=device)

    # Resume weights
    if args.resume is not None and args.resume != "None":
        load_flag, resume_desc = load_pretrained_component(videoMol, args.resume, "videomol", consistency=False)
        assert load_flag, "error in loading pretrained model {}.".format(args.resume)
        print("[resume description] {}".format(resume_desc))

    model_params_num = cal_torch_model_params(videoMol, unit="M")
    print(videoMol)
    print("videoMol: {}".format(model_params_num))

    # Loss and optimizer
    optim_params = filter(lambda x: x.requires_grad, videoMol.parameters())
    optimizer = SGD(optim_params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # model with gpu
    if len(device_ids) > 1:
        print("starting multi-gpu.")
        videoMol = torch.nn.DataParallel(videoMol, device_ids=device_ids).cuda()
    else:
        videoMol = videoMol.to(device)

    ########################## train
    results = {'highest_valid': min_value,
               'final_train': min_value,
               'final_test': min_value,
               'highest_train': min_value}

    early_stop = 0
    patience = 30
    for epoch in range(args.start_epoch, args.epochs):
        tqdm_train_desc, tqdm_eval_train_desc, tqdm_eval_val_desc, tqdm_eval_test_desc = get_tqdm_desc(args.dataset, epoch)

        if isinstance(videoMol, RegMeanStdWrapper):
            videoMol.set_train()
        train_one_epoch(model=videoMol, optimizer=optimizer, data_loader=train_dataloader, criterion=criterion,
                        device=device, epoch=epoch, task_type=args.task_type, tqdm_desc=tqdm_train_desc)

        # evaluate
        if isinstance(videoMol, RegMeanStdWrapper):
            videoMol.set_eval()
        train_loss, train_results, train_data_dict = evaluate_on_video(model=videoMol, data_loader=train_dataloader,
                                                                       criterion=criterion, device=device,
                                                                       n_frame=args.n_frame, task_type=args.task_type,
                                                                       return_data_dict=True,
                                                                       inverse_scale_mean_std=inverse_scale_mean_std,
                                                                       tqdm_desc=tqdm_eval_train_desc)
        val_loss, val_results, val_data_dict = evaluate_on_video(model=videoMol, data_loader=valid_dataloader,
                                                                 criterion=criterion, device=device,
                                                                 n_frame=args.n_frame, task_type=args.task_type,
                                                                 return_data_dict=True, tqdm_desc=tqdm_eval_val_desc)
        test_loss, test_results, test_data_dict = evaluate_on_video(model=videoMol, data_loader=test_dataloader,
                                                                    criterion=criterion, device=device,
                                                                    n_frame=args.n_frame, task_type=args.task_type,
                                                                    return_data_dict=True,
                                                                    tqdm_desc=tqdm_eval_test_desc)

        train_result = train_results[eval_metric.upper()]
        valid_result = val_results[eval_metric.upper()]
        test_result = test_results[eval_metric.upper()]

        print({"dataset": args.dataset, "epoch": epoch, "Loss": train_loss, 'Train': train_result,
               'Validation': valid_result, 'Test': test_result})

        if is_left_better_right(train_result, results['highest_train'], standard=valid_select):
            results['highest_train'] = train_result

        if is_left_better_right(valid_result, results['highest_valid'], standard=valid_select):
            results['highest_valid'] = valid_result
            results['final_train'] = train_result
            results['final_test'] = test_result

            if args.save_finetune_ckpt == 1:
                save_finetune_ckpt(videoMol, optimizer, round(train_loss, 4), epoch, args.log_dir, "valid_best",
                                   lr_scheduler=None, result_dict=results)

            early_stop = 0
        else:
            early_stop += 1
            if early_stop > patience:
                break

    print("results: {}\n".format(results))


if __name__ == '__main__':
    args = parse_args()
    main(args)
