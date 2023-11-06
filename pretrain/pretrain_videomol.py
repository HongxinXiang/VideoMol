import sys
sys.path.append("/root/xianghongxin/work/videomol/")
import os
import glob
from argparse import ArgumentParser
import sys
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from utils.public_utils import fix_train_random_seed, cal_torch_model_params
from model.frame.frame_model import AxisClassifier, RotationClassifier, AngleClassifier, ChemicalClassifier
from dataloader.data_utils import simple_transforms_for_train, transforms_for_eval, load_video_data_list_1
from dataloader.dataset import PretrainFrameDataset
from utils.splitter import split_train_val_idx
import numpy as np
from pathlib import Path
from utils.logger import Logger
from pretrain.pretrain_utils import train_one_epoch, evaluate_on_frame
from loss.losses import SupConLoss
from model.model_utils import save_checkpoint, load_checkpoint
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from model.base.predictor import FramePredictor
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


def setup_DDP_mp(init_method, local_rank, rank, world_size, backend="nccl", verbose=False):
    if sys.platform == "win32":
        backend = "gloo"
    # If the OS is Windows or macOS, use gloo instead of nccl
    dist.init_process_group(backend=backend, init_method=init_method, world_size=world_size, rank=rank)
    # set distributed device
    device = torch.device("cuda:{}".format(local_rank))
    if verbose:
        print("Using device: {}".format(device))
        print(f"local rank: {local_rank}, global rank: {rank}, world size: {world_size}")
    return device


def parse_args():
    # Parse arguments
    parser = ArgumentParser(description='PyTorch Implementation of VideoMol')

    # basic
    parser.add_argument('--dataroot', type=str, default="../datasets/pre-training/", help='data root, e.g. ../datasets/pre-training/')
    parser.add_argument('--dataset', type=str, default="head-100-video-20w-224x224", help='dataset name, e.g. head-100-video-20w-224x224')
    parser.add_argument('--label_column_name', type=str, default="label", help='column name of label')
    parser.add_argument('--workers', default=2, type=int, help='number of data loading workers (default: 2)')

    # ddp
    parser.add_argument("--nodes", default=1, type=int, help="number of nodes for distributed training")
    parser.add_argument("--ngpus_per_node", default=2, type=int, help="number of GPUs per node for distributed training")
    parser.add_argument("--dist-url", default="tcp://127.0.0.1:12355", type=str, help="url used to set up distributed training")
    parser.add_argument("--node_rank", default=0, type=int, help="node rank for distributed training")

    # model params
    parser.add_argument('--model_name', type=str, default="swin_s3_tiny_224", help='model name')
    parser.add_argument('--n_chemical_classes', type=int, default=100, help='number of clusters')
    parser.add_argument('--n_frame', type=int, default=60, help='the number of frame')
    parser.add_argument('--mode', type=str, default="mean", choices=["mean", "sum", "sub"], help='')

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument('--weighted_loss', action='store_false', help='add regularization for multi-task loss')

    # train
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42) to split dataset')
    parser.add_argument('--runseed', type=int, default=2021, help='random seed to run model (default: 2021)')
    parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts) (default: 0)')
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument('--imageSize', type=int, default=224, help='the height / width of the input image to network')
    parser.add_argument('--temperature', type=float, default=0.1, help="temperature required by contrastive loss")
    parser.add_argument('--base_temperature', type=float, default=0.1, help="temperature required by contrastive loss")
    parser.add_argument("--resume", type=str, default=None, help='Resume training from a path of checkpoint')
    parser.add_argument("--validation-split", type=float, default=0.1, help="Validation split")
    parser.add_argument('--n_ckpt_save', default=1, type=int, help='save a checkpoint every n epochs, n_ckpt_save=0: no save')
    parser.add_argument('--n_batch_step_optim', default=1, type=int, help='update model parameters every n batches')

    # log
    parser.add_argument('--log_dir', default='./logs/pretrain-frame/', help='path to log')

    # Parse arguments
    return parser.parse_args()


def print_only_rank0(text, logger=None):
    log = print if logger is None else logger.info
    if dist.get_rank() == 0:
        log(text)


def is_rank0():
    return dist.get_rank() == 0


def main(local_rank, ngpus_per_node, args):

    args.local_rank = local_rank
    args.rank = args.node_rank * ngpus_per_node + local_rank

    device = setup_DDP_mp(init_method=args.dist_url, local_rank=args.local_rank, rank=args.rank,
                          world_size=args.world_size, verbose=True)

    # fix random seeds
    fix_train_random_seed(args.runseed)

    # initializing logger
    args.log_dir = args.log_dir / Path(args.dataset) / Path(args.model_name) / Path("seed" + str(args.seed))
    args.tb_dir = os.path.join(args.log_dir, "tb")
    log_filename = "logs.log"
    log_path = args.log_dir / Path(log_filename)
    try:
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
    except:
        pass
    logMaster = Logger(str(log_path))
    logger = logMaster.get_logger('main')
    print_only_rank0("run command: " + " ".join(sys.argv), logger=logger)
    print_only_rank0("log_dir: {}".format(args.log_dir), logger=logger)

    ########################## load dataset
    train_transforms = simple_transforms_for_train(resize=args.imageSize, mean_std=(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    valid_transforms = transforms_for_eval(resize=(args.imageSize, args.imageSize),
                                           img_size=(args.imageSize, args.imageSize),
                                           mean_std=(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))

    # Load dataset
    print_only_rank0("load dataset", logger=logger)
    fast_read_path = os.path.join(args.dataroot, args.dataset, "processed", "load_video_data_list_1_{}_{}.npz".format(args.dataset, args.label_column_name))
    if os.path.exists(fast_read_path):
        data = np.load(fast_read_path)
        video_index_list, video_root_list, video_label_list = data['video_index_list'], data['video_root_list'], data['video_label_list']
    else:
        video_index_list, video_root_list, video_label_list = load_video_data_list_1(args.dataroot, args.dataset, label_column_name=args.label_column_name)
        np.savez(fast_read_path, video_index_list=video_index_list, video_root_list=video_root_list, video_label_list=video_label_list)
    video_index_list, video_root_list, video_labels = np.array(video_index_list), np.array(video_root_list), np.array(video_label_list)
    n_chemical_classes = args.n_chemical_classes  # n_cluster

    # split train/valid
    print_only_rank0("split train/valid", logger=logger)
    idx = list(range(len(video_index_list)))
    train_idx, valid_idx = split_train_val_idx(idx, shuffle=True, stratify=None, sort=True, seed=42,
                                               frac_train=1 - args.validation_split, frac_valid=args.validation_split)

    debug = False
    if debug:
        train_idx = train_idx[:300]
        valid_idx = valid_idx[:300]

    train_video_index_list, train_video_root_list, train_video_labels = video_index_list[train_idx], video_root_list[train_idx], video_labels[train_idx]
    valid_video_index_list, valid_video_root_list, valid_video_labels = video_index_list[valid_idx], video_root_list[valid_idx], video_labels[valid_idx]

    train_frameDataset = PretrainFrameDataset(train_video_root_list, video_labels=train_video_labels, video_indexs=train_video_index_list, transforms=train_transforms, n_frame=args.n_frame, ret_index=True)
    valid_frameDataset = PretrainFrameDataset(valid_video_root_list, video_labels=valid_video_labels, video_indexs=valid_video_index_list, transforms=valid_transforms, n_frame=args.n_frame, ret_index=True)

    # initialize data loader
    batch_size = args.batch // args.world_size
    train_sampler = DistributedSampler(train_frameDataset, shuffle=True)
    valid_sampler = DistributedSampler(valid_frameDataset, shuffle=False)
    train_loader = DataLoader(train_frameDataset, batch_size=batch_size, sampler=train_sampler, num_workers=args.workers, pin_memory=True)
    valid_loader = DataLoader(valid_frameDataset, batch_size=batch_size // 2, sampler=valid_sampler, num_workers=args.workers, pin_memory=True)

    ########################## load model
    videoMol = FramePredictor(model_name=args.model_name, head_arch="none", num_tasks=None, pretrained=True)
    num_features = videoMol.in_features

    axisClassifier = AxisClassifier(in_features=num_features)
    rotationClassifier = RotationClassifier(in_features=num_features)
    angleClassifier = AngleClassifier(in_features=num_features, num_tasks=args.n_frame//3-1)
    chemicalClassifier = ChemicalClassifier(in_features=num_features, num_tasks=n_chemical_classes)

    model_params_num = cal_torch_model_params(videoMol, unit="M")
    axisClassifier_params_num = cal_torch_model_params(axisClassifier, unit="M")
    rotationClassifier_params_num = cal_torch_model_params(rotationClassifier, unit="M")
    angleClassifier_params_num = cal_torch_model_params(angleClassifier, unit="M")
    chemicalClassifier_params_num = cal_torch_model_params(chemicalClassifier, unit="M")

    print_only_rank0("videoMol: {}".format(model_params_num), logger=logger)
    print_only_rank0("axisClassifier: {}".format(axisClassifier_params_num), logger=logger)
    print_only_rank0("rotationClassifier: {}".format(rotationClassifier_params_num), logger=logger)
    print_only_rank0("angleClassifier: {}".format(angleClassifier_params_num), logger=logger)
    print_only_rank0("chemicalClassifier: {}".format(chemicalClassifier_params_num), logger=logger)

    # Loss and optimizer
    optim_params = [{"params": videoMol.parameters()},
                    {"params": axisClassifier.parameters()},
                    {"params": rotationClassifier.parameters()},
                    {"params": angleClassifier.parameters()},
                    {"params": chemicalClassifier.parameters()}]
    optimizer = SGD(optim_params, momentum=args.momentum, lr=args.lr, weight_decay=args.weight_decay)
    criterionCL = SupConLoss(temperature=args.temperature, base_temperature=args.base_temperature, contrast_mode='all')  # mutual information
    criterionCE = nn.CrossEntropyLoss()

    # lr scheduler
    lr_scheduler = None

    # Resume weights
    if args.resume is not None:
        flag, resume_desc = load_checkpoint(args.resume, videoMol, axisClassifier, rotationClassifier, angleClassifier,
                                            chemicalClassifier, optimizer=None, lr_scheduler=None, logger=logger)
        args.start_epoch = int(resume_desc['epoch'])
        assert flag, "error in loading pretrained model {}.".format(args.resume)
        print_only_rank0("[resume description] {}".format(resume_desc), logger=logger)

    # model with DDP
    print_only_rank0("starting DDP.", logger=logger)
    videoMol = DDP(videoMol.to(device), device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
    axisClassifier = DDP(axisClassifier.to(device), device_ids=[local_rank], output_device=local_rank,
                         broadcast_buffers=False)
    rotationClassifier = DDP(rotationClassifier.to(device), device_ids=[local_rank], output_device=local_rank,
                             broadcast_buffers=False)
    angleClassifier = DDP(angleClassifier.to(device), device_ids=[local_rank], output_device=local_rank,
                          broadcast_buffers=False)
    chemicalClassifier = DDP(chemicalClassifier.to(device), device_ids=[local_rank], output_device=local_rank,
                             broadcast_buffers=False)

    ########################## train
    best_loss = np.Inf
    for epoch in range(args.start_epoch, args.epochs):
        # [*] set sampler
        train_loader.sampler.set_epoch(epoch)
        valid_loader.sampler.set_epoch(epoch)

        train_dict = train_one_epoch(frame_model=videoMol, axisClassifier=axisClassifier,
                                     rotationClassifier=rotationClassifier, angleClassifier=angleClassifier,
                                     chemicalClassifier=chemicalClassifier, optimizer=optimizer,
                                     data_loader=train_loader, criterionCL=criterionCL,
                                     criterionCE=criterionCE, device=device, epoch=epoch,
                                     lr_scheduler=lr_scheduler, args=args, mode=args.mode,
                                     weighted_loss=args.weighted_loss, is_ddp=True)

        print_only_rank0(str(train_dict), logger=logger)

        evaluate_valid_results = evaluate_on_frame(videoMol, axisClassifier, rotationClassifier, angleClassifier,
                                                   chemicalClassifier, valid_loader, criterionCL, criterionCE,
                                                   device, mode=args.mode, args=args)
        print_only_rank0("[valid evaluation] epoch: {} | {}".format(epoch, evaluate_valid_results), logger=logger)

        # save model
        model_dict = {"videomol": videoMol, "axisClassifier": axisClassifier,
                      "rotationClassifier": rotationClassifier, "angleClassifier": angleClassifier,
                      "chemicalClassifier": chemicalClassifier}
        optimizer_dict = {"optimizer": optimizer}
        lr_scheduler_dict = {"lr_scheduler": lr_scheduler} if lr_scheduler is not None else None

        cur_loss = train_dict["total_loss"]

        # save best model
        if is_rank0() and best_loss > cur_loss:
            files2remove = glob.glob(os.path.join(args.log_dir, "ckpts", "best_epoch*"))
            for _i in files2remove:
                os.remove(_i)
            best_loss = cur_loss
            best_pre = "best_epoch={}_loss={:.2f}".format(epoch, best_loss)
            save_checkpoint(model_dict, optimizer_dict, lr_scheduler_dict,
                            train_dict, epoch, save_path=os.path.join(args.log_dir, "ckpts"),
                            name_pre=best_pre, name_post="", logger=logger)

        if is_rank0() and args.n_ckpt_save > 0 and epoch % args.n_ckpt_save == 0:
            ckpt_pre = "ckpt_epoch={}_loss={:.2f}".format(epoch, cur_loss)
            save_checkpoint(model_dict, optimizer_dict, lr_scheduler_dict,
                            train_dict, epoch, save_path=os.path.join(args.log_dir, "ckpts"),
                            name_pre=ckpt_pre, name_post="", logger=logger)


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    args = parse_args()
    args.world_size = args.ngpus_per_node * args.nodes
    mp.spawn(main, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
