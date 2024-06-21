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



## Data processing

### 1. Generating 3D conformation for your data

We use RDKiT[<sup>1</sup>](#ref1) to generate a 3D conformation for each molecule if the molecules do not have a conformation. Here is the code snippet to generate the 3D conformation:

<details>
  <summary>Click here for the code!</summary>

```python
def generate_3d_comformer(smiles, sdf_save_path, mmffVariant="MMFF94", randomSeed=0, maxIters=5000, increment=2, optim_count=10, save_force=False):
    count = 0
    while count < optim_count:
        try:
            m = Chem.MolFromSmiles(smiles)
            m3d = Chem.AddHs(m)
            if save_force:
                try:
                    AllChem.EmbedMolecule(m3d, randomSeed=randomSeed)
                    res = AllChem.MMFFOptimizeMolecule(m3d, mmffVariant=mmffVariant, maxIters=maxIters)
                    m3d = Chem.RemoveHs(m3d)
                except:
                    m3d = Chem.RemoveHs(m3d)
                    print("forcing saving molecule which can't be optimized ...")
                    mol2sdf(m3d, sdf_save_path)
            else:
                AllChem.EmbedMolecule(m3d, randomSeed=randomSeed)
                res = AllChem.MMFFOptimizeMolecule(m3d, mmffVariant=mmffVariant, maxIters=maxIters)
                m3d = Chem.RemoveHs(m3d)
        except Exception as e:
            traceback.print_exc()
        if res == 1:
            maxIters = maxIters * increment
            count += 1
            continue
        mol2sdf(m3d, sdf_save_path)
    if save_force:
        print("forcing saving molecule without convergence ...")
        mol2sdf(m3d, sdf_save_path)
```

</details>



### 2. Rendering molecular video

We use PyMOL[<sup>2</sup>](#ref2) to render each frame of molecular video, which is a user-sponsored molecular visualization system on an open-source foundation, maintained and distributed by Schrödinger. You can get it for free from [the link](https://pymol.org/2/).

Here is the PyMOL script to get the molecular frame, you can run it in the PyMOL command:

<details>
<summary>Click here for the code!</summary>

```bash
sdf_filepath=demo.sdf
rotate_direction=x  # x,y,z
rotate=30  # any angle from 0~360
save_img_path=demo_frame.png
load $sdf_filepath;bg_color white;hide (hydro);set stick_ball,on;set stick_ball_ratio,3.5;set stick_radius,0.15;set sphere_scale,0.2;set valence,1;set valence_mode,0;set valence_size, 0.1;rotate $rotate_direction, $rotate;save $save_img_path;quit;
```

</details>



## 🔥Pre-training

#### 1. Preparing dataset for pre-training

You can download provided [pretraining data](https://drive.google.com/file/d/1RkzYcJUQUtp5sqQis-mQvzurSznLkh2N/view?usp=sharing) and push it into the folder `datasets/pre-training/`. 

Note that larger pre-training data means more storage space is required. You have free access to all pre-trained datasets in [PCQM4Mv2](https://ogb.stanford.edu/docs/lsc/pcqm4mv2/)[<sup>3</sup>](#ref3), of which we use the first 2 million molecules.



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

The downstream datasets can be accessed in following table:

| Name                          | Download link                                                | Description                   |
| ----------------------------- | ------------------------------------------------------------ | ----------------------------- |
| KinomeScan.zip                | [GoogleDrive](https://drive.google.com/file/d/1Q6yZEhB9ATNZxjZB9tR6zB_sm6B49aaM/view?usp=sharing) | 10 kinase datasets            |
| kinases.zip                   | [OneDrive](https://1drv.ms/u/s!Atau0ecyBQNTgRhTW7aoX_ecTFLt?e=Ab7WyI) | 10 GPCR datasets              |
| SARS-CoV-2_REDIAL-2020.tar.gz | [OneDrive](https://1drv.ms/u/s!Atau0ecyBQNTgRmtGGcJQpKBrU3o?e=idhTHJ) | 11 SARS-CoV-2 datasets        |
| MPP                           | [OneDrive](https://1drv.ms/f/s!Atau0ecyBQNTgRrf1iE-eogd17M-?e=m7so1Q) | Molecular property prediction |

Please download all data listed above and push them into the folder `datasets/fine-tuning/`



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



## Reproducing our results

1.  Download the dataset from the "preparing downstream datasets" section.

2.  Click [this link](https://github.com/HongxinXiang/VideoMol/scripts) to view the reproducibility guide.



The code for other comparison methods can be accessed through [this link](https://1drv.ms/f/s!Atau0ecyBQNTgTd736-8RPWEXSVt?e=DkOyw2).



# Reference

<div id="ref1">[1] Landrum G. RDKit: A software suite for cheminformatics, computational chemistry, and predictive modeling[J]. Greg Landrum, 2013, 8: 31.</div>

<div id="ref2">[2] DeLano W L. Pymol: An open-source molecular graphics tool[J]. CCP4 Newsl. Protein Crystallogr, 2002, 40(1): 82-92.</div>

<div id="ref3">[3] Hu W, Fey M, Ren H, et al. Ogb-lsc: A large-scale challenge for machine learning on graphs[J]. arXiv preprint arXiv:2103.09430, 2021.</div>

