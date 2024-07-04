import os.path
import random
from collections import defaultdict

import numpy as np
import pandas as pd
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.transforms_factory import create_transform
from torch import nn
from torchvision import transforms
from tqdm import tqdm


class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)


def simple_transforms_for_train(resize=224, mean_std=None, p=0.3):
    if mean_std is None:
        mean, std = IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    else:
        mean, std = mean_std[0], mean_std[1]
    img_transforms = transforms.Compose([transforms.CenterCrop(resize), transforms.RandomGrayscale(p=p),
                                         RandomApply(transforms.ColorJitter(brightness=[0.6, 1.4], contrast=[0.6, 1.4], saturation=[0.6, 1.4]), p=p),
                                         RandomApply(transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)), p=p),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=mean, std=std)])
    return img_transforms


def aug2_transforms_for_train(resize=224, mean_std=None, p=0.2, rotation=False):
    if mean_std is None:
        mean, std = IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    else:
        mean, std = mean_std[0], mean_std[1]
    img_transforms = transforms.Compose([transforms.CenterCrop(resize), transforms.RandomHorizontalFlip(),
                                         transforms.RandomGrayscale(p), transforms.RandomRotation(degrees=360),
                                         transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    return img_transforms


def simple_transforms_no_aug(mean_std=None):
    if mean_std is None:
        mean, std = IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    else:
        mean, std = mean_std[0], mean_std[1]
    img_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    return img_transforms


def imagemol_transforms_for_train(resize=224, mean_std=None, p=0.2, rotation=False):
    if mean_std is None:
        mean, std = IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    else:
        mean, std = mean_std[0], mean_std[1]
    img_transforms = transforms.Compose([transforms.CenterCrop(resize), transforms.RandomHorizontalFlip(),
                                         transforms.RandomGrayscale(p), transforms.RandomRotation(degrees=360),
                                         transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    return img_transforms


def transforms_for_train(config, is_training=True):
    return create_transform(**config, is_training=is_training)


def transforms_for_eval(resize=(224, 224), img_size=(224, 224), mean_std=None):
    if mean_std is None:
        mean, std = IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    else:
        mean, std = mean_std[0], mean_std[1]
    return transforms.Compose([
        transforms.Resize(resize),
        transforms.RandomCrop(img_size),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


def check_num_frame_of_video(video_path_list, n_frame):
    for idx, frame_path_list in enumerate(video_path_list):
        assert len(frame_path_list) == n_frame, \
            "The frame number of video {} is {}, not equal to the expected {}"\
                .format(idx, len(frame_path_list), n_frame)


def load_video_data_list(dataroot, dataset, video_type="processed", label_column_name="label", video_dir_name="video", csv_suffix=""):
    csv_file_path = os.path.join(dataroot, dataset, video_type, "{}{}.csv".format(dataset, csv_suffix))
    df = pd.read_csv(csv_file_path)
    columns = df.columns.tolist()
    assert "index" in columns and label_column_name in columns
    video_root = os.path.join(dataroot, dataset, video_type, video_dir_name)

    video_index_list = df["index"].tolist()
    video_label_list = df[label_column_name].apply(lambda x: str(x).split(' ')).tolist()
    video_path_list = []

    for video_index in tqdm(video_index_list, desc="load_video_data_list"):
        root1 = os.path.join(video_root, str(video_index))
        video_path = []
        for filename in os.listdir(root1):
            video_path.append(os.path.join(root1, filename))
        # sort video_path (0.png, 1.png, 2.png, ...)
        video_path.sort(key=lambda x: int(os.path.split(x)[1].split('.')[0]))
        video_path_list.append(video_path)

    return video_index_list, video_path_list, video_label_list


def load_video_data_list_1(dataroot, dataset, video_type="processed", label_column_name="label", video_dir_name="video", csv_suffix=""):

    csv_file_path = os.path.join(dataroot, dataset, video_type, "{}{}.csv".format(dataset, csv_suffix))
    df = pd.read_csv(csv_file_path)
    columns = df.columns.tolist()
    assert "index" in columns and label_column_name in columns
    video_root = os.path.join(dataroot, dataset, video_type, video_dir_name)

    video_index_list = df["index"].tolist()
    video_label_list = df[label_column_name].tolist()
    video_root_list = []

    for video_index in tqdm(video_index_list, desc="load_video_data_list"):
        root1 = os.path.join(video_root, str(video_index))
        video_root_list.append(root1)
    return video_index_list, video_root_list, video_label_list


def load_random_single_frame_data_list(dataroot, dataset, n_frame, video_type="processed", label_column_name="label",
                                       video_dir_name="video", csv_suffix="", random=True, seed=2022):
    csv_file_path = os.path.join(dataroot, dataset, video_type, "{}{}.csv".format(dataset, csv_suffix))
    df = pd.read_csv(csv_file_path)
    columns = df.columns.tolist()
    assert "index" in columns and label_column_name in columns
    video_root = os.path.join(dataroot, dataset, video_type, video_dir_name)

    video_index_list = df["index"].tolist()
    video_label_list = df[label_column_name].apply(lambda x: str(x).split(' ')).tolist()
    video_path_list = []

    for video_index in video_index_list:
        root1 = os.path.join(video_root, str(video_index))
        video_path = []
        filenames = os.listdir(root1)
        assert len(filenames) == n_frame
        for filename in filenames:
            video_path.append(os.path.join(root1, filename))
        # sort video_path (0.png, 1.png, 2.png, ...)
        video_path.sort(key=lambda x: int(os.path.split(x)[1].split('.')[0]))
        video_path_list.append(video_path)

    # sample single frame for each video
    rng = np.random.RandomState(seed)
    video_path_list = np.array(video_path_list)
    n_samples = len(video_label_list)
    assert video_path_list.shape == (n_samples, n_frame)
    if random:
        random_index = rng.randint(low=0, high=n_frame, size=(n_samples))
    else:
        random_index = np.array([0] * n_samples)
    random_single_frame_path_list = video_path_list[range(n_samples), random_index].tolist()

    return video_index_list, random_single_frame_path_list, video_label_list


def get_axis_index(n_frame):
    x_start, x_end = 0, n_frame // 3  # x axis: [0, n_frame // 3)
    y_start, y_end = n_frame // 3, n_frame // 3 * 2  # y axis: [n_frame // 3, n_frame // 3 * 2)
    z_start, z_end = n_frame // 3 * 2, n_frame // 3 * 3  # x axis: [n_frame // 3 * 2, n_frame // 3 * 3)
    x_axis_index = list(range(x_start, x_end))
    y_axis_index = list(range(y_start, y_end))
    z_axis_index = list(range(z_start, z_end))
    return x_axis_index, y_axis_index, z_axis_index


def extract_video_filename_data(video_root, filenames):
    data = defaultdict(list)
    for filename in filenames:
        video_index, png_name = os.path.split(filename)
        data[int(video_index)].append(f"{video_root}/{filename}")
    video_index_list = list(data.keys())
    video_path_list = np.array(list(data.values()))
    assert len(video_path_list.shape) == 2
    return video_index_list, video_path_list


def load_multi_view_split_file(video_root, split_path):
    npz_data = np.load(split_path)
    train_idx, val_idx, test_idx = npz_data['idx_train'], npz_data['idx_val'], npz_data['idx_test']
    train_video_labels, valid_video_labels, test_video_labels = npz_data['labels_train'], npz_data['labels_val'], npz_data['labels_test']
    train_video_index_list, train_video_path_list = extract_video_filename_data(video_root, filenames=npz_data['filename_train'])
    valid_video_index_list, valid_video_path_list = extract_video_filename_data(video_root, filenames=npz_data['filename_val'])
    test_video_index_list, test_video_path_list = extract_video_filename_data(video_root, filenames=npz_data['filename_test'].flatten())

    return train_idx, val_idx, test_idx, \
        train_video_index_list, valid_video_index_list, test_video_index_list, \
        train_video_path_list, valid_video_path_list, test_video_path_list, \
        train_video_labels, valid_video_labels, test_video_labels
