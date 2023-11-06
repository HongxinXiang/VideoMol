import sys
import os
import torch
import numpy as np
from sklearn import metrics
from tqdm import tqdm
from utils.evaluate import metric as utils_evaluate_metric
from utils.evaluate import metric_multitask as utils_evaluate_metric_multitask
from utils.evaluate import metric_reg as utils_evaluate_metric_reg
from utils.evaluate import metric_reg_multitask as utils_evaluate_metric_reg_multitask
from model.base.predictor import RegMeanStdWrapper


def metric(y_true, y_pred, y_prob):
    auc = metrics.roc_auc_score(y_true, y_prob)
    return {
        "ROCAUC": auc,
    }


# save checkpoint
def save_finetune_ckpt(model, optimizer, loss, epoch, save_path, filename_pre, lr_scheduler=None, result_dict=None, logger=None):
    log = logger.info if logger is not None else print
    model_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
    lr_scheduler = None if lr_scheduler is None else lr_scheduler.state_dict()
    state = {
            'epoch': epoch,
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


def train_one_epoch(model, optimizer, data_loader, criterion, device, epoch, task_type, tqdm_desc=""):
    assert task_type in ["classification", "regression"]
    if isinstance(model, RegMeanStdWrapper):
        model.set_train()

    model.train()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, desc=tqdm_desc)
    for step, data in enumerate(data_loader):
        if len(data) == 2:
            images, labels = data
        elif len(data) == 3:
            images, labels, _ = data
        else:
            raise Exception("error in len(data)={}".format(len(data)))

        images, labels = images.to(device), labels.to(device)
        if isinstance(model, RegMeanStdWrapper):
            labels = (labels - model.targets_mean) / model.targets_std

        sample_num += images.shape[0]

        pred = model(images)
        labels = labels.view(pred.shape).to(torch.float64)
        if task_type == "classification":
            is_valid = labels != -1
            loss_mat = criterion(pred.double(), labels)
            loss_mat = torch.where(is_valid, loss_mat,
                                   torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            loss = torch.sum(loss_mat) / torch.sum(is_valid)
        elif task_type == "regression":
            loss = criterion(pred.double(), labels)

        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}".format(epoch, accu_loss.item() / (step + 1))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1)


@torch.no_grad()
def evaluate_on_video(model, data_loader, criterion, device, n_frame, task_type="classification",
                      inverse_scale_mean_std=None, return_data_dict=False, tqdm_desc=""):

    assert task_type in ["classification", "regression"]
    if isinstance(model, RegMeanStdWrapper):
        model.set_eval()

    model.eval()

    accu_loss = torch.zeros(1).to(device)

    y_scores, y_true, y_pred, y_prob, frame_indexs = [], [], [], [], []
    sample_num = 0
    data_loader = tqdm(data_loader, desc=tqdm_desc)
    for step, data in enumerate(data_loader):
        assert len(data) == 3
        images, labels, indexs = data
        if inverse_scale_mean_std is not None:
            mean, std = inverse_scale_mean_std
            labels = labels * std + mean
        images, labels = images.to(device), labels.to(device)
        sample_num += images.shape[0]

        with torch.no_grad():
            pred = model(images)
            labels = labels.view(pred.shape).to(torch.float64)
            if task_type == "classification":
                is_valid = labels != -1
                loss_mat = criterion(pred.double(), labels)
                loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
                loss = torch.sum(loss_mat) / torch.sum(is_valid)
            elif task_type == "regression":
                loss = criterion(pred.double(), labels)
            accu_loss += loss.detach()
            data_loader.desc = "{}; loss: {:.3f}".format(tqdm_desc, accu_loss.item() / (step + 1))

        y_true.append(labels.view(pred.shape))
        y_scores.append(pred)
        frame_indexs.append(indexs)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

    frame_indexs = torch.cat(frame_indexs, dim=0).numpy()
    n_samples, n_tasks = len(frame_indexs), y_true.shape[1]
    argsort_frame_indexs = np.argsort(frame_indexs)

    assert np.unique(y_true[argsort_frame_indexs].reshape(-1, n_frame, n_tasks), axis=1).shape[1] == 1
    y_video_true, y_video_scores = y_true[argsort_frame_indexs].reshape(-1, n_frame, n_tasks)[:, 0], y_scores[
        argsort_frame_indexs].reshape(-1, n_frame, n_tasks).mean(axis=1)

    if y_true.shape[1] == 1:
        if task_type == "classification":
            y_video_pro = torch.sigmoid(torch.Tensor(y_video_scores))
            y_video_pred = torch.where(y_video_pro > 0.5, torch.Tensor([1]), torch.Tensor([0])).numpy()

            video_level_results = utils_evaluate_metric(y_video_true, y_video_pred, y_video_pro, empty=-1)
            results = video_level_results
            if return_data_dict:
                data_dict = {"video_level_results": video_level_results}
                return accu_loss.item() / (step + 1), results, data_dict
            else:
                return accu_loss.item() / (step + 1), results

        elif task_type == "regression":
            video_level_results = utils_evaluate_metric_reg(y_video_true, y_video_scores)
            if return_data_dict:
                data_dict = {"video_level_results": video_level_results}
                return accu_loss.item() / (step + 1), video_level_results, data_dict
            else:
                return accu_loss.item() / (step + 1), video_level_results

    elif y_true.shape[1] > 1:  # multi-task
        if task_type == "classification":
            y_video_pro = torch.sigmoid(torch.Tensor(y_video_scores))
            y_video_pred = torch.where(y_video_pro > 0.5, torch.Tensor([1]), torch.Tensor([0])).numpy()

            video_level_results = utils_evaluate_metric_multitask(y_video_true, y_video_pred, y_video_pro, num_tasks=y_true.shape[1], empty=-1)
            if return_data_dict:
                data_dict = {"video_level_results": video_level_results}
                return accu_loss.item() / (step + 1), video_level_results, data_dict
            else:
                return accu_loss.item() / (step + 1), video_level_results

        elif task_type == "regression":
            video_level_results = utils_evaluate_metric_reg_multitask(y_video_true, y_video_scores, num_tasks=y_video_true.shape[1])
            if return_data_dict:
                data_dict = {"video_level_results": video_level_results}
                return accu_loss.item() / (step + 1), video_level_results, data_dict
            else:
                return accu_loss.item() / (step + 1), video_level_results
    else:
        raise Exception("error in the number of task.")


def get_video_filename(video_path_list):
    video_filename_list = []
    for video_paths in video_path_list:
        if isinstance(video_paths, str):
            video_path = video_paths
            video_dir, filename = os.path.split(video_path)
            _, video_index = os.path.split(video_dir)
            video_filename_list.append("{}/{}".format(video_index, filename))
        else:
            video_filenames = []
            for video_path in video_paths:
                video_dir, filename = os.path.split(video_path)
                _, video_index = os.path.split(video_dir)
                video_filenames.append("{}/{}".format(video_index, filename))
            video_filename_list.append(video_filenames)
    return video_filename_list


def get_scaffold_regression_metainfo():
    task_metainfo = {
        "esol": {
            "mean": -2.8668758314855878,  # np.mean(data['labels_train']), np.std(data['labels_train'])
            "std": 2.066724108076815,
            "target_name": "logSolubility",
        },
        "freesolv": {
            "mean": -3.2594736842105267,  # np.mean(data['labels_train']), np.std(data['labels_train'])
            "std": 3.2775297233608893,
            "target_name": "freesolv",
        },
        "lipophilicity": {"mean": 2.162160714285714, "std": 1.2115398399123027, "target_name": "lipophilicity"},  # np.mean(data['labels_train']), np.std(data['labels_train'])

        "qm7": {
            "mean": -1553.3112591508054,  # np.mean(data['labels_train']), np.std(data['labels_train'])
            "std": 228.87222455180435,
            "target_name": "u0_atom",
        },
        "qm8": {
            "mean": [
                 0.2215273,
                 0.25086629,
                 0.02201264,
                 0.04294198,
                 0.21755504,
                 0.24307655,
                 0.01984034,
                 0.03324616,
                 0.21725409,
                 0.24517239,
                 0.02185256,
                 0.03711905
            ],
            "std": [
                0.04386131,
                0.03429066,
                0.05253202,
                0.07290261,
                0.04749187,
                0.03993779,
                0.05098404,
                0.0610024,
                0.04427453,
                0.03580219,
                0.05550889,
                0.06744566
            ],
            "target_name": [
                "E1-CC2",
                "E2-CC2",
                "f1-CC2",
                "f2-CC2",
                "E1-PBE0",
                "E2-PBE0",
                "f1-PBE0",
                "f2-PBE0",
                "E1-CAM",
                "E2-CAM",
                "f1-CAM",
                "f2-CAM",
            ],
        },
        "qm9": {
            "mean": [-0.24139767,  0.01100605,  0.25240365],
            "std": [0.02250266, 0.04683212, 0.04720357],
            "target_name": ["homo", "lumo", "gap"],
        },
    }
    return task_metainfo
