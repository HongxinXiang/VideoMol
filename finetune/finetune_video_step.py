import os
import sys
from argparse import ArgumentParser

import pandas as pd
import torch
import torch.nn as nn
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.optim import SGD
from torch.utils.data import DataLoader

from dataloader.data_utils import simple_transforms_for_train, transforms_for_eval, simple_transforms_no_aug
from dataloader.dataset import FrameDataset
from finetune.train_utils import evaluate_on_video
from finetune.train_utils import get_scaffold_regression_metainfo
from model.base.predictor import FramePredictor, RegMeanStdWrapper
from model.model_utils import load_pretrained_component
from utils.public_utils import fix_train_random_seed, setup_device, cal_torch_model_params, get_tqdm_desc, \
    is_left_better_right
from utils.splitter import *
from utils.train_utils import load_smiles


def parse_args():
    # Parse arguments
    parser = ArgumentParser(description='PyTorch Implementation of VideoMol for Fine-Tuning with Step Mode')

    # basic
    parser.add_argument('--dataroot', type=str, default="../datasets/fine-tuning/", help='data root')
    parser.add_argument('--dataset', type=str, default="bbbp", help='dataset name, e.g. bbbp, clintox, ...')
    parser.add_argument('--label_column_name', type=str, default="label", help='column name of label')
    parser.add_argument('--video_dir_name', type=str, default="video-224x224", help='directory name of video')
    parser.add_argument('--gpu', type=str, default="0", help='GPUs of CUDA_VISIBLE_DEVICES, e.g. 0,1,2,3')
    parser.add_argument('--ngpu', type=int, default=8, help='number of GPUs to use')
    parser.add_argument('--workers', default=8, type=int, help='number of data loading workers (default: 8)')

    # optimizer
    parser.add_argument("--warmup_rate", type=float, default=1e-3, help="warmup rate")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay")

    # train
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42) to split dataset')
    parser.add_argument('--runseed', type=int, default=2022, help='random seed to run model (default: 2022)')
    parser.add_argument('--split', default="random", type=str,
                        choices=['split_file', 'random', 'stratified', 'scaffold', 'random_scaffold',
                                 'balanced_scaffold'], help='regularization of classification loss')
    parser.add_argument('--split_path', type=str, help='e.g. ./dataset/mpp/BBBP/scaffold.npy')
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs")
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts) (default: 0)')
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--resume", type=str, default=None, help='Resume training from a path of checkpoint')
    parser.add_argument('--arch', default="arch1", type=str, choices=['arch1', 'arch2', 'arch3', 'arch4', 'arch5'],
                        help='prediction classifier (fine-tuning stage)')
    parser.add_argument('--imageSize', type=int, default=224, help='the height / width of the input image to network')
    parser.add_argument('--model_name', type=str, default="vit_small_patch16_224", help='model name')
    parser.add_argument('--n_frame', type=int, default=60, help='the number of frame')
    parser.add_argument('--close_image_aug', action='store_true', default=False,
                        help='whether to use data augmentation')
    parser.add_argument('--task_type', type=str, default="classification", choices=["classification", "regression"],
                        help='task type')
    parser.add_argument('--save_finetune_ckpt', type=int, default=1, choices=[0, 1],
                        help='1 represents saving best ckpt, 0 represents no saving best ckpt')
    parser.add_argument('--reg_mean_std', default="none", type=str, choices=['none', 'self'], help='')
    parser.add_argument("--eval_step", type=int, default=5, help="")

    # params
    parser.add_argument("--dropout", type=float, default=0.5, help="")
    parser.add_argument("--activation_fn", type=str, default="gelu", help="",
                        choices=["relu", "gelu", "tanh", "softplus", "linear"])

    parser.add_argument('--log_dir', default='./logs/finetune_video_step', help='path to log')

    # Parse arguments
    return parser.parse_args()


# save checkpoint
def save_finetune_ckpt(model, optimizer, loss, epoch, step, save_path, filename_pre, lr_scheduler=None,
                       result_dict=None, logger=None):
    log = logger.info if logger is not None else print
    model_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
    lr_scheduler = None if lr_scheduler is None else lr_scheduler.state_dict()
    state = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model_cpu,
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler,
        'loss': loss,
        'result_dict': result_dict
    }
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        log("Directory {} is created.".format(save_path))

    filename = '{}/{}.pth'.format(save_path, filename_pre)
    torch.save(state, filename)
    log('model has been saved as {}'.format(filename))


def load_views_data_list(dataroot, dataset, filenames, video_type="processed", label_column_name="label",
                         video_dir_name="video", csv_suffix=""):
    csv_file_path = os.path.join(dataroot, dataset, video_type, "{}{}.csv".format(dataset, csv_suffix))
    df = pd.read_csv(csv_file_path)
    columns = df.columns.tolist()
    assert "index" in columns and label_column_name in columns
    video_root = f"{dataroot}/{dataset}/{video_type}/{video_dir_name}"

    video_index_list = df["index"].tolist()
    video_label_list = df[label_column_name].apply(lambda x: str(x).split(' ')).tolist()
    video_path_list = []

    for video_index in tqdm(video_index_list, desc="load_video_data_list"):
        root1 = f"{video_root}/{video_index}"
        video_path = [f"{root1}/{filename}" for filename in filenames]
        video_path_list.append(video_path)

    return video_index_list, video_path_list, video_label_list


def main(args):
    ########################## basic
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    args.csv_file = f"{args.dataroot}/{args.dataset}/processed/{args.dataset}_processed_ac.csv"

    # gpus
    device, device_ids = setup_device(args.ngpu)

    # fix random seeds
    fix_train_random_seed(args.runseed)

    ######################### basic information of data
    stat_mean, stat_std = None, None
    if args.task_type == "classification":
        eval_metric = "rocauc"
        valid_select = "max"
        min_value = -np.inf
        criterion = nn.BCEWithLogitsLoss(reduction="none")
    elif args.task_type == "regression":
        if args.dataset == "qm7" or args.dataset == "qm8" or args.dataset == "qm9":
            eval_metric = "mae"
            criterion = nn.SmoothL1Loss()
        else:
            eval_metric = "rmse"
            criterion = nn.MSELoss()
        valid_select = "min"
        min_value = np.inf

        if args.reg_mean_std == "self":
            task_metainfo = get_scaffold_regression_metainfo()
        else:
            task_metainfo = None
        if task_metainfo is not None and args.dataset in task_metainfo:
            # for regression task, pre-compute mean and std
            stat_mean = task_metainfo[args.dataset]["mean"]
            stat_std = task_metainfo[args.dataset]["std"]
    else:
        raise Exception("param {} is not supported".format(args.task_type))
    print("eval_metric: {}".format(eval_metric))

    ########################## load dataset
    if args.close_image_aug:
        train_transforms = simple_transforms_no_aug(mean_std=(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
        eval_transforms = simple_transforms_no_aug(mean_std=(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    else:
        train_transforms = simple_transforms_for_train(resize=args.imageSize,
                                                       mean_std=(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
        eval_transforms = transforms_for_eval(resize=(args.imageSize, args.imageSize),
                                              img_size=(args.imageSize, args.imageSize),
                                              mean_std=(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))

    filenames = [f"{item}.png" for item in range(60)]
    if args.dataset in ["hiv", "qm8", "qm9"]:
        filenames = filenames[::12]
        print("use args.n_frame=5")
        args.n_frame = 5

    video_dir_name = args.video_dir_name
    video_index_list, video_path_list, video_label_list = load_views_data_list(args.dataroot, args.dataset, filenames,
                                                                               label_column_name=args.label_column_name,
                                                                               video_dir_name=video_dir_name,
                                                                               csv_suffix="_processed_ac")
    video_index_list, video_path_list, video_labels = np.array(video_index_list), np.array(video_path_list), np.array(
        video_label_list).reshape(len(video_label_list), -1)
    if args.task_type == "classification":
        video_labels = video_labels.astype(int)
    elif args.task_type == "regression":
        video_labels = video_labels.astype(float)
    num_videos, num_tasks = video_labels.shape

    # split train/valid/test
    if args.split == "split_file":
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
        train_idx, val_idx, test_idx = scaffold_split_train_val_test(list(range(0, num_videos)), smiles,
                                                                     frac_train=0.8, frac_valid=0.1, frac_test=0.1)

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
        raise NotImplementedError

    train_video_index_list, train_video_path_list, train_video_labels = video_index_list[train_idx], video_path_list[
        train_idx], video_labels[train_idx]
    valid_video_index_list, valid_video_path_list, valid_video_labels = video_index_list[val_idx], video_path_list[
        val_idx], video_labels[val_idx]
    test_video_index_list, test_video_path_list, test_video_labels = video_index_list[test_idx], video_path_list[
        test_idx], video_labels[test_idx]

    train_frameDataset = FrameDataset(train_video_path_list, video_labels=train_video_labels,
                                      video_indexs=train_video_index_list, transforms=train_transforms,
                                      n_frame=args.n_frame, ret_index=True)
    valid_frameDataset = FrameDataset(valid_video_path_list, video_labels=valid_video_labels,
                                      video_indexs=valid_video_index_list, transforms=eval_transforms,
                                      n_frame=args.n_frame, ret_index=True)
    test_frameDataset = FrameDataset(test_video_path_list, video_labels=test_video_labels,
                                     video_indexs=test_video_index_list, transforms=eval_transforms,
                                     n_frame=args.n_frame, ret_index=True)
    train_dataloader = DataLoader(train_frameDataset, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                                  pin_memory=True)
    valid_dataloader = DataLoader(valid_frameDataset, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                                  pin_memory=True)
    test_dataloader = DataLoader(test_frameDataset, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                                 pin_memory=True)

    ########################## load model
    print("add mlp [{}] for downstream task".format(args.arch))
    videoMol = FramePredictor(model_name=args.model_name, head_arch=args.arch, num_tasks=num_tasks, pretrained=False,
                              head_arch_params={"inner_dim": None, "dropout": args.dropout,
                                                "activation_fn": args.activation_fn})

    if stat_mean and stat_std:
        videoMol = RegMeanStdWrapper(model=videoMol, stat_mean=stat_mean, stat_std=stat_std, evaluate=False,
                                     device=device)

    # Resume weights
    if args.resume is not None and args.resume != "None":
        load_flag, resume_desc = load_pretrained_component(videoMol, args.resume, "frame_model", consistency=False,
                                                           logger=None)
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
               'highest_train': min_value,
               'highest_valid_desc': None,
               "final_train_desc": None,
               "final_test_desc": None,
               'data_split': {
                   "train_idx": train_idx,
                   "val_idx": val_idx,
                   "test_idx": test_idx
               }}

    early_stop = 0
    patience = 30
    for epoch in range(args.start_epoch, args.epochs):
        tqdm_train_desc, tqdm_eval_train_desc, tqdm_eval_val_desc, tqdm_eval_test_desc = get_tqdm_desc(args.dataset,
                                                                                                       epoch)
        if isinstance(videoMol, RegMeanStdWrapper):
            videoMol.set_train()
        videoMol.train()

        # ================= train one epoch
        assert args.task_type in ["classification", "regression"]

        accu_loss = torch.zeros(1).to(device)  # 累计损失
        optimizer.zero_grad()

        sample_num = 0
        len_data = len(train_dataloader)
        stop_step = len_data // args.eval_step
        train_dataloader = tqdm(train_dataloader, desc=tqdm_train_desc)
        for step, data in enumerate(train_dataloader):
            if len(data) == 2:
                images, labels = data
            elif len(data) == 3:
                images, labels, _ = data
            else:
                raise Exception("error in len(data)={}".format(len(data)))

            images, labels = images.to(device), labels.to(device)
            if isinstance(videoMol, RegMeanStdWrapper):
                labels = (labels - videoMol.targets_mean) / videoMol.targets_std

            sample_num += images.shape[0]

            pred = videoMol(images)
            labels = labels.view(pred.shape).to(torch.float64)
            if args.task_type == "classification":
                is_valid = labels != -1
                loss_mat = criterion(pred.double(), labels)
                loss_mat = torch.where(is_valid, loss_mat,
                                       torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
                loss = torch.sum(loss_mat) / torch.sum(is_valid)
            elif args.task_type == "regression":
                loss = criterion(pred.double(), labels)
            else:
                raise Exception

            loss.backward()
            accu_loss += loss.detach()

            train_dataloader.desc = "[train epoch {}] loss: {:.3f}".format(epoch, accu_loss.item() / (step + 1))

            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss)
                sys.exit(1)

            optimizer.step()
            optimizer.zero_grad()

            # ================= eval one epoch
            if step != 0 and (step % stop_step == 0 or step == len_data - 1):
                loss = accu_loss.item() / (step + 1)

                eval_train = False
                if not eval_train:
                    train_loss, train_results, train_data_dict = loss, {eval_metric.upper(): "no evaluate"}, None
                else:
                    train_loss, train_results, train_data_dict = evaluate_on_video(model=videoMol,
                                                                                   data_loader=train_dataloader,
                                                                                   criterion=criterion,
                                                                                   device=device,
                                                                                   n_frame=args.n_frame,
                                                                                   task_type=args.task_type,
                                                                                   return_data_dict=True,
                                                                                   tqdm_desc=tqdm_eval_train_desc)
                val_loss, val_results, val_data_dict = evaluate_on_video(model=videoMol,
                                                                         data_loader=valid_dataloader,
                                                                         criterion=criterion, device=device,
                                                                         n_frame=args.n_frame,
                                                                         task_type=args.task_type,
                                                                         return_data_dict=True,
                                                                         tqdm_desc=tqdm_eval_val_desc)
                test_loss, test_results, test_data_dict = evaluate_on_video(model=videoMol,
                                                                            data_loader=test_dataloader,
                                                                            criterion=criterion, device=device,
                                                                            n_frame=args.n_frame,
                                                                            task_type=args.task_type,
                                                                            return_data_dict=True,
                                                                            tqdm_desc=tqdm_eval_test_desc)

                train_result = train_results[eval_metric.upper()]
                valid_result = val_results[eval_metric.upper()]
                test_result = test_results[eval_metric.upper()]

                print({"dataset": args.dataset, "epoch": epoch, "step": step, "Loss": train_loss,
                       'Train': train_result, 'Validation': valid_result, 'Test': test_result})

                if eval_train:
                    if is_left_better_right(train_result, results['highest_train'], standard=valid_select):
                        results['highest_train'] = train_result

                if is_left_better_right(valid_result, results['highest_valid'], standard=valid_select):
                    results['highest_valid'] = valid_result
                    results['final_train'] = train_result
                    results['final_test'] = test_result

                    results['highest_valid_desc'] = val_results
                    results['final_train_desc'] = train_results
                    results['final_test_desc'] = test_results

                    if args.save_finetune_ckpt == 1:
                        save_finetune_ckpt(videoMol, optimizer, round(train_loss, 4), epoch, step, args.log_dir,
                                           "valid_best",
                                           lr_scheduler=None, result_dict=results, logger=None)

                    early_stop = 0
                else:
                    early_stop += 1
                    if early_stop > patience:
                        break

                if isinstance(videoMol, RegMeanStdWrapper):
                    videoMol.set_train()
                videoMol.train()

    print("results: {}\n".format(results))


if __name__ == '__main__':
    args = parse_args()

    print(args.log_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    main(args)
