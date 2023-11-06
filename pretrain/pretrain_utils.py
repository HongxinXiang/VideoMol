import sys
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch.distributed.nn import all_gather
from tqdm import tqdm

from dataloader.data_utils import get_axis_index


def generate_direction_aware_label(frame1_index, frame2_index, n_frame, device):
    assert n_frame % 3 == 0
    np_frame1_index = frame1_index.cpu().numpy()
    np_frame2_index = frame2_index.cpu().numpy()

    # axis
    x_axis_index, y_axis_index, z_axis_index = get_axis_index(n_frame=n_frame)
    x_index = np.where(np.isin(np_frame1_index, x_axis_index))
    y_index = np.where(np.isin(np_frame1_index, y_axis_index))
    z_index = np.where(np.isin(np_frame1_index, z_axis_index))
    aixs_label = np.zeros(shape=(np_frame1_index.shape[0], ))
    aixs_label[x_index] = 0
    aixs_label[y_index] = 1
    aixs_label[z_index] = 2
    aixs_label = torch.Tensor(aixs_label).to(device).long()

    # rotation
    rotation_label = np_frame2_index - np_frame1_index
    rotation_label[rotation_label > 0] = 0  # clockwise
    rotation_label[rotation_label < 0] = 1  # counterclockwise
    rotation_label = torch.Tensor(rotation_label).to(device).long()

    # angle
    angle_label = abs(np_frame2_index - np_frame1_index) - 1  # Does not include pairs without rotation
    angle_label = torch.Tensor(angle_label).to(device).long()

    return aixs_label, rotation_label, angle_label


def train_one_epoch(frame_model, axisClassifier, rotationClassifier, angleClassifier, chemicalClassifier,
                    optimizer, data_loader, criterionCL, criterionCE, device, epoch, mode="concat",
                    weighted_loss=False, lr_scheduler=None, args=None, is_ddp=False):

    frame_model.train()
    axisClassifier.train()
    rotationClassifier.train()
    angleClassifier.train()
    chemicalClassifier.train()

    accu_loss = torch.zeros(1).to(device)
    accu_CL_loss = torch.zeros(1).to(device)
    accu_direction_loss = torch.zeros(1).to(device)
    accu_chemical_loss = torch.zeros(1).to(device)

    accu_axis_loss = torch.zeros(1).to(device)
    accu_roration_loss = torch.zeros(1).to(device)
    accu_angle_loss = torch.zeros(1).to(device)

    accu_chem1_loss = torch.zeros(1).to(device)
    accu_chem2_loss = torch.zeros(1).to(device)

    optimizer.zero_grad()

    train_dict = {}
    data_loader = tqdm(data_loader, total=len(data_loader))
    for step, (frames, labels, video_indexs, frame_indexs) in enumerate(data_loader):
        if frames.shape[0] <= 1:
            continue
        frame1, frame2 = frames[:, 0].to(device), frames[:, 1].to(device)
        frame1_index, frame2_index = frame_indexs[:, 0].to(device).long(), frame_indexs[:, 1].to(device).long()
        labels, video_indexs, frame_indexs = labels.to(device).long(), video_indexs.to(device), frame_indexs.to(device)

        # forward
        feat1, feat2 = frame_model(frame1), frame_model(frame2)

        # frame-level contrastive learning (Frames belonging to the same video index should be as similar as possible)
        if is_ddp:
            all_feat1 = torch.cat(all_gather(feat1), dim=0)
            all_feat2 = torch.cat(all_gather(feat2), dim=0)
            all_video_indexs = torch.cat(all_gather(video_indexs), dim=0)
            L_CL = criterionCL(all_feat1, all_feat2, labels=all_video_indexs)
        else:
            L_CL = criterionCL(feat1, feat2, labels=video_indexs)

        # direction-aware
        if mode == "concat":
            pair_feat = torch.cat([feat1, feat2], dim=1)
        elif mode == "mean":
            pair_feat = (feat1 + feat2) / 2
        elif mode == "sub":
            pair_feat = (feat1 - feat2) / 2
        elif mode == "sum":
            pair_feat = feat1 + feat2
        else:
            raise Exception("mode {} is undefined".format(mode))

        aixs_true, rotation_true, angle_true = generate_direction_aware_label(frame1_index, frame2_index, args.n_frame,
                                                                              device)
        axis_logits, rotation_logits, angle_logits = axisClassifier(pair_feat), rotationClassifier(
            pair_feat), angleClassifier(pair_feat)

        # direction loss
        L_axis = criterionCE(axis_logits, aixs_true)
        L_rotation = criterionCE(rotation_logits, rotation_true)
        L_angle = criterionCE(angle_logits, angle_true)
        L_direction = L_axis + L_rotation + L_angle

        # chemical-aware loss
        chemical_logits1 = chemicalClassifier(feat1)
        chemical_logits2 = chemicalClassifier(feat2)
        L_chem1 = criterionCE(chemical_logits1, labels)
        L_chem2 = criterionCE(chemical_logits2, labels)
        L_chemical = (L_chem1 + L_chem2) * 0.5

        # backward
        if weighted_loss:
            loss = L_CL + L_direction + L_chemical
            weight_L_CL = (L_CL / loss * 5).detach()
            weight_L_axis = (L_axis / loss * 5).detach()
            weight_L_rotation = (L_rotation / loss * 5).detach()
            weight_L_angle = (L_angle / loss * 5).detach()
            weight_L_chemical = (L_chemical / loss * 5).detach()
            weighted_L_direction = weight_L_axis * L_axis + weight_L_rotation * L_rotation + weight_L_angle * L_angle
            weighted_loss = weight_L_CL * L_CL + weighted_L_direction + weight_L_chemical * L_chemical
            weighted_loss.backward()
        else:
            loss = L_CL + L_direction + L_chemical
            loss.backward()

        # logger
        accu_loss += loss.detach()
        accu_CL_loss += (L_CL).detach()
        accu_direction_loss += (L_direction).detach()
        accu_chemical_loss += L_chemical.detach()
        accu_axis_loss += (L_axis).detach()
        accu_roration_loss += (L_rotation).detach()
        accu_angle_loss += (L_angle).detach()
        accu_chem1_loss += (L_chem1).detach() * 0.5
        accu_chem2_loss += (L_chem2).detach() * 0.5

        data_loader.desc = "[train epoch {}] total loss: {:.3f}; " \
                           "CL loss: {:.3f}; direction loss: {:.3f}; chemical loss: {:.3f}; " \
                           "[accu_axis_loss: {:.3f}, accu_roration_loss: {:.3f}, accu_angle_loss: {:.3f}; " \
                           "accu_chem1_loss: {:.3f}, accu_chem2_loss: {:.3f}]". \
            format(epoch, accu_loss.item() / (step + 1), accu_CL_loss.item() / (step + 1),
                   accu_direction_loss.item() / (step + 1), accu_chemical_loss.item() / (step + 1),
                   accu_axis_loss.item() / (step + 1), accu_roration_loss.item() / (step + 1),
                   accu_angle_loss.item() / (step + 1), accu_chem1_loss.item() / (step + 1),
                   accu_chem2_loss.item() / (step + 1))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        if step % args.n_batch_step_optim == 0:
            optimizer.step()
            optimizer.zero_grad()

        train_dict = {
            "step": (step + 1) + len(data_loader) * epoch,
            "epoch": epoch + (step + 1) / len(data_loader),
            "total_loss": accu_loss.item() / (step + 1),
            "CL_loss": accu_CL_loss.item() / (step + 1),
            "direction_loss": accu_direction_loss.item() / (step + 1),
            "chemical_loss": accu_chemical_loss.item() / (step + 1),
            "axis_loss": accu_axis_loss.item() / (step + 1),
            "roration_loss": accu_roration_loss.item() / (step + 1),
            "angle_loss": accu_angle_loss.item() / (step + 1),
            "chem1_loss": accu_chem1_loss.item() / (step + 1),
            "chem2_loss": accu_chem2_loss.item() / (step + 1),
        }

    # Update learning rates
    if lr_scheduler is not None:
        lr_scheduler.step()

    return train_dict


def evaluate_on_frame(frame_model, axisClassifier, rotationClassifier, angleClassifier, chemicalClassifier,
                      data_loader, criterionCL, criterionCE, device, mode="concat", args=None):
    frame_model.eval()
    axisClassifier.eval()
    rotationClassifier.eval()
    angleClassifier.eval()
    chemicalClassifier.eval()

    evaluate_results = {
        "loss": defaultdict(int),  # default 0
        "accuracy": defaultdict(int)  # default 0
    }

    all_step = len(data_loader)
    data_loader = tqdm(data_loader, total=all_step)
    for step, (frames, labels, video_indexs, frame_indexs) in enumerate(data_loader):
        if frames.shape[0] <= 1:
            continue
        frame1, frame2 = frames[:, 0].to(device), frames[:, 1].to(device)
        frame1_index, frame2_index = frame_indexs[:, 0].to(device).long(), frame_indexs[:, 1].to(device).long()
        labels, video_indexs, frame_indexs = labels.to(device).long(), video_indexs.to(device), frame_indexs.to(device)

        # forward
        feat1, feat2 = frame_model(frame1), frame_model(frame2)

        # frame-level contrastive learning
        L_CL = criterionCL(feat1, feat2, labels=video_indexs)

        # direction-aware
        if mode == "concat":
            pair_feat = torch.cat([feat1, feat2], dim=1)
        elif mode == "mean":
            pair_feat = (feat1 + feat2) / 2
        elif mode == "sub":
            pair_feat = (feat1 - feat2) / 2
        elif mode == "sum":
            pair_feat = feat1 + feat2
        else:
            raise Exception("mode {} is undefined".format(mode))
        aixs_true, rotation_true, angle_true = generate_direction_aware_label(frame1_index, frame2_index, args.n_frame, device)
        axis_logits, rotation_logits, angle_logits = axisClassifier(pair_feat), rotationClassifier(pair_feat), angleClassifier(pair_feat)

        L_axis = criterionCE(axis_logits, aixs_true)
        L_rotation = criterionCE(rotation_logits, rotation_true)
        L_angle = criterionCE(angle_logits, angle_true)
        L_direction = L_axis + L_rotation + L_angle

        # chemical-aware
        chemical_logits1 = chemicalClassifier(feat1)
        chemical_logits2 = chemicalClassifier(feat2)
        L_chem1 = criterionCE(chemical_logits1, labels)
        L_chem2 = criterionCE(chemical_logits2, labels)
        L_chemical = (L_chem1 + L_chem2) * 0.5

        # log information
        loss_piar = [('L_CL', L_CL), ('L_direction', L_direction), ('L_chemical', L_chemical),
                     ('L_axis', L_axis), ('L_rotation', L_rotation), ('L_angle', L_angle),
                     ('L_chem1', L_chem1), ('L_chem2', L_chem1)]
        for name, item in loss_piar:
            evaluate_results['loss'][name] += item.cpu().item() / all_step

        evaluate_package = ('axis', axis_logits, aixs_true), ('rotation', rotation_logits, rotation_true), ('angle', angle_logits, angle_true), ('chemical1', chemical_logits1, labels), ('chemical2', chemical_logits2, labels)
        for name, y_logit, y_true in evaluate_package:
            y_pred = np.argmax(y_logit.detach().cpu().numpy(), axis=1)
            evaluate_results['accuracy'][name] += accuracy_score(y_true.cpu().numpy(), y_pred) / all_step

        evaluate_results['loss'] = dict(evaluate_results['loss'])
        evaluate_results['accuracy'] = dict(evaluate_results['accuracy'])

    return evaluate_results

