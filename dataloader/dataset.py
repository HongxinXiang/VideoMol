import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from dataloader.data_utils import check_num_frame_of_video, get_axis_index


class FrameDataset(Dataset):
    def __init__(self, video_path_list, video_labels, video_indexs, transforms, n_frame=16, ret_index=False, args=None):
        check_num_frame_of_video(video_path_list, n_frame)

        self.args = args
        self.frame_path_list = np.concatenate(video_path_list)
        self.frame_labels = np.repeat(video_labels, n_frame, axis=0)
        self.frame_indexs = np.repeat(video_indexs, n_frame, axis=0)
        assert len(self.frame_path_list) == len(self.frame_labels) == len(self.frame_indexs)

        self.total_video = len(video_path_list)
        self.total_frame = len(video_path_list) * n_frame
        self.transforms = transforms
        self.n_frame = n_frame
        self.ret_index = ret_index

    def get_frame(self, index):
        filename = self.frame_path_list[index]
        img = Image.open(filename).convert('RGB')
        return img

    def __getitem__(self, index):
        frame = self.get_frame(index)

        if self.transforms is not None:
            frame = self.transforms(frame)

        if self.ret_index:
            return frame, self.frame_labels[index], self.frame_indexs[index]
        else:
            return frame, self.frame_labels[index]

    def __len__(self):
        return self.total_frame


class PretrainFrameDataset(Dataset):
    def __init__(self, video_root_list, video_labels, video_indexs, transforms=None, n_frame=16, ret_index=False, args=None):
        # check_num_frame_of_video(video_path_list, n_frame)
        self.args = args
        self.video_root_list = video_root_list
        self.video_labels = video_labels
        self.video_indexs = video_indexs

        self.total_video = len(self.video_root_list)
        self.total_frame = len(self.video_root_list) * n_frame
        self.transforms = transforms
        self.n_frame = n_frame
        self.ret_index = ret_index

    def get_pair_frame(self, index):
        video_root = self.video_root_list[index]
        pair_idx = self.sample_pair_frames_from_same_axis()
        frame_path_list = os.path.join(video_root, f"{pair_idx[0]}.png"), os.path.join(video_root, f"{pair_idx[1]}.png")
        pair_frame = [Image.open(frame_path).convert('RGB') for frame_path in frame_path_list]
        return pair_frame, pair_idx

    def __getitem__(self, index):
        pair_frame, pair_idx = self.get_pair_frame(index)

        if self.transforms is not None:
            pair_frame = list(map(lambda img: self.transforms(img).unsqueeze(0), pair_frame))
            pair_frame = torch.cat(pair_frame)

        if self.ret_index:
            return pair_frame, self.video_labels[index], self.video_indexs[index], np.array(pair_idx)
        else:
            return pair_frame, self.video_labels[index]

    def __len__(self):
        return self.total_video

    def sample_pair_frames_from_same_axis(self):
        assert self.n_frame % 3 == 0
        x_axis_index, y_axis_index, z_axis_index = get_axis_index(self.n_frame)
        axis_index = [x_axis_index, y_axis_index, z_axis_index]
        selected_axis = random.sample(range(len(axis_index)), 1)[0]
        pair_idx = random.sample(axis_index[selected_axis], 2)
        return pair_idx
