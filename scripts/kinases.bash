source activate videomol
cd your_command_path

python finetune-video.py --dataroot ./datasets --dataset 5HT1A --batch 8 --lr 1e-05 --epoch 30 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type regression --seed 0 --runseed 0
python finetune-video.py --dataroot ./datasets --dataset 5HT1A --batch 8 --lr 1e-05 --epoch 30 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type regression --seed 0 --runseed 1
python finetune-video.py --dataroot ./datasets --dataset 5HT1A --batch 8 --lr 1e-05 --epoch 30 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type regression --seed 0 --runseed 2

python finetune-video.py --dataroot ./datasets --dataset 5HT2A --batch 16 --lr 0.0005 --epoch 30 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type regression --seed 0 --runseed 1
python finetune-video.py --dataroot ./datasets --dataset 5HT2A --batch 16 --lr 0.0005 --epoch 30 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type regression --seed 0 --runseed 2
python finetune-video.py --dataroot ./datasets --dataset 5HT2A --batch 16 --lr 0.0005 --epoch 30 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type regression --seed 0 --runseed 0

python finetune-video.py --dataroot ./datasets --dataset AA1R --batch 8 --lr 1e-05 --epoch 30 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type regression --seed 0 --runseed 0
python finetune-video.py --dataroot ./datasets --dataset AA1R --batch 8 --lr 1e-05 --epoch 30 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type regression --seed 0 --runseed 1
python finetune-video.py --dataroot ./datasets --dataset AA1R --batch 8 --lr 1e-05 --epoch 30 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type regression --seed 0 --runseed 2

python finetune-video.py --dataroot ./datasets --dataset AA2AR --batch 8 --lr 1e-05 --epoch 30 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type regression --seed 0 --runseed 0
python finetune-video.py --dataroot ./datasets --dataset AA2AR --batch 8 --lr 1e-05 --epoch 30 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type regression --seed 0 --runseed 1
python finetune-video.py --dataroot ./datasets --dataset AA2AR --batch 8 --lr 1e-05 --epoch 30 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type regression --seed 0 --runseed 2

python finetune-video.py --dataroot ./datasets --dataset AA3R --batch 32 --lr 0.0005 --epoch 30 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type regression --seed 0 --runseed 0
python finetune-video.py --dataroot ./datasets --dataset AA3R --batch 32 --lr 0.0005 --epoch 30 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type regression --seed 0 --runseed 1
python finetune-video.py --dataroot ./datasets --dataset AA3R --batch 32 --lr 0.0005 --epoch 30 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type regression --seed 0 --runseed 2

python finetune-video.py --dataroot ./datasets --dataset CNR2 --batch 8 --lr 1e-05 --epoch 30 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type regression --seed 0 --runseed 0
python finetune-video.py --dataroot ./datasets --dataset CNR2 --batch 8 --lr 1e-05 --epoch 30 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type regression --seed 0 --runseed 1
python finetune-video.py --dataroot ./datasets --dataset CNR2 --batch 8 --lr 1e-05 --epoch 30 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type regression --seed 0 --runseed 2

python finetune-video.py --dataroot ./datasets --dataset DRD2 --batch 8 --lr 1e-05 --epoch 30 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type regression --seed 0 --runseed 0
python finetune-video.py --dataroot ./datasets --dataset DRD2 --batch 8 --lr 1e-05 --epoch 30 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type regression --seed 0 --runseed 1
python finetune-video.py --dataroot ./datasets --dataset DRD2 --batch 8 --lr 1e-05 --epoch 30 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type regression --seed 0 --runseed 2

python finetune-video.py --dataroot ./datasets --dataset DRD3 --batch 16 --lr 0.0005 --epoch 30 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type regression --seed 0 --runseed 0
python finetune-video.py --dataroot ./datasets --dataset DRD3 --batch 16 --lr 0.0005 --epoch 30 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type regression --seed 0 --runseed 1
python finetune-video.py --dataroot ./datasets --dataset DRD3 --batch 16 --lr 0.0005 --epoch 30 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type regression --seed 0 --runseed 2

python finetune-video.py --dataroot ./datasets --dataset HRH3 --batch 8 --lr 1e-05 --epoch 30 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type regression --seed 0 --runseed 0
python finetune-video.py --dataroot ./datasets --dataset HRH3 --batch 8 --lr 1e-05 --epoch 30 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type regression --seed 0 --runseed 1
python finetune-video.py --dataroot ./datasets --dataset HRH3 --batch 8 --lr 1e-05 --epoch 30 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type regression --seed 0 --runseed 2

python finetune-video.py --dataroot ./datasets --dataset OPRM --batch 8 --lr 0.0001 --epoch 30 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type regression --seed 0 --runseed 0
python finetune-video.py --dataroot ./datasets --dataset OPRM --batch 8 --lr 0.0001 --epoch 30 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type regression --seed 0 --runseed 1
python finetune-video.py --dataroot ./datasets --dataset OPRM --batch 8 --lr 0.0001 --epoch 30 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type regression --seed 0 --runseed 2
