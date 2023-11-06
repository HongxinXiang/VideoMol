import os
import pandas as pd
import numpy as np


def load_filenames_and_labels_multitask(image_folder, txt_file, task_type="classification"):
    assert task_type in ["classification", "regression"]
    df = pd.read_csv(txt_file)
    index = df["index"].values.astype(int)
    labels = np.array(df.label.apply(lambda x: str(x).split(' ')).tolist())
    labels = labels.astype(int) if task_type == "classification" else labels.astype(float)
    names = [os.path.join(image_folder, str(item)+".png") for item in index]
    assert len(index) == labels.shape[0] == len(names)
    return names, labels


def get_datasets(dataset, dataroot, data_type="raw"):
    assert data_type in ["raw", "processed"]

    image_folder = os.path.join(dataroot, "{}/{}/image-224x224/".format(dataset, data_type))
    txt_file = os.path.join(dataroot, "{}/{}/{}_processed_ac.csv".format(dataset, data_type, dataset))

    assert os.path.isdir(image_folder), "{} is not a directory.".format(image_folder)
    assert os.path.isfile(txt_file), "{} is not a file.".format(txt_file)

    return image_folder, txt_file
