# VideoMol

Prediction of drug targets and properties using a molecular video-derived foundation model



## News!

**[2023/11/05]** Repository installation completed.



## Environments

#### 1. GPU environment

CUDA 11.6

Ubuntu 18.04



#### 2. create conda environment

```bash
# create conda env
conda create -n videomol python=3.9
conda activate videomol

# install environment
pip install rdkit -i https://mirrors.tuna.tsinghua.edu.cn
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install timm==0.6.12
pip install tensorboard
pip install scikit-learn
pip install setuptools==59.5.0
pip install pandas
```



## Pre-training

#### 1. Preparing dataset for pre-training

You can download provided [pretraining data](https://drive.google.com/file/d/1RkzYcJUQUtp5sqQis-mQvzurSznLkh2N/view?usp=sharing) and push it into the folder `datasets/pre-training/`.



#### 2. Pre-train VideoMol

Usage:

```bash
usage: pretrain_videomol.py [-h] [--dataroot DATAROOT] [--dataset DATASET]
                            [--label_column_name LABEL_COLUMN_NAME]
                            [--workers WORKERS] [--nodes NODES]
                            [--ngpus_per_node NGPUS_PER_NODE]
                            [--dist-url DIST_URL] [--node_rank NODE_RANK]
                            [--model_name MODEL_NAME]
                            [--n_chemical_classes N_CHEMICAL_CLASSES]
                            [--n_frame N_FRAME] [--mode {mean,sum,sub}]
                            [--lr LR] [--momentum MOMENTUM]
                            [--weight-decay WEIGHT_DECAY] [--weighted_loss]
                            [--seed SEED] [--runseed RUNSEED]
                            [--start_epoch START_EPOCH] [--epochs EPOCHS]
                            [--batch BATCH] [--imageSize IMAGESIZE]
                            [--temperature TEMPERATURE]
                            [--base_temperature BASE_TEMPERATURE]
                            [--resume RESUME]
                            [--validation-split VALIDATION_SPLIT]
                            [--n_ckpt_save N_CKPT_SAVE]
                            [--n_batch_step_optim N_BATCH_STEP_OPTIM]
                            [--log_dir LOG_DIR]
```

For example, you can use the following command to pretrain VideoMol:

```bash
python pretrain_videomol.py \
--nodes=1 \
--ngpus_per_node=1 \
--model_name vit_small_patch16_224 \
--mode sub \
--n_batch_step_optim 1 \
--epochs 100 \
--batch 8 \
--weighted_loss \
--lr 1e-2 \
--ngpu 1 \
--workers 16 \
--dataroot ../datasets/pre-training/ \
--dataset video-1000-224x224 \
--label_column_name k100 \
--log_dir ./experiments/videomol/pretrain_videomol/
```



## Fine-tuning

####  1. preparing pretrained videomol

Download [pre-trained model](https://drive.google.com/file/d/1TitrL3ed5Wko_xJxornnXFp4DRJLW6ya/view?usp=sharing) and push it into the folder `ckpts/`



#### 2. preparing downstream datasets

Download [GPCRs](https://drive.google.com/file/d/1Q6yZEhB9ATNZxjZB9tR6zB_sm6B49aaM/view?usp=sharing) datasets and push them into the folder `datasets/fine-tuning/`



#### 3. fine-tuning with pretrained videomol

Usage:

```bash
usage: finetune_video.py [-h] [--dataroot DATAROOT] [--dataset DATASET]
                         [--label_column_name LABEL_COLUMN_NAME]
                         [--video_dir_name VIDEO_DIR_NAME] [--gpu GPU]
                         [--ngpu NGPU] [--workers WORKERS] [--lr LR]
                         [--momentum MOMENTUM] [--weight-decay WEIGHT_DECAY]
                         [--seed SEED] [--runseed RUNSEED]
                         [--split SPLIT]
                         [--split_path SPLIT_PATH] [--epochs EPOCHS]
                         [--start_epoch START_EPOCH] [--batch BATCH]
                         [--resume RESUME] [--arch {arch1,arch2,arch3}]
                         [--imageSize IMAGESIZE] [--model_name MODEL_NAME]
                         [--n_frame N_FRAME] [--close_image_aug]
                         [--task_type {classification,regression}]
                         [--save_finetune_ckpt {0,1}] [--log_dir LOG_DIR]
```

For examples:

```bash
python --dataroot ../datasets/fine-tuning/KinomeScan/ --dataset BTK --epochs 10
```



# Reference



