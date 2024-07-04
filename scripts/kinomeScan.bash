source activate videomol
cd your_command_path


python finetune_video.py --dataroot ./datasets --dataset BTK --batch 32 --lr 0.1 --epoch 30 --split balanced_scaffold --arch arch1 --resume ./resumes/videomol.pth --task_type classification --seed 0 --runseed 2 --image_aug_name simple
python finetune_video.py --dataroot ./datasets --dataset BTK --batch 8 --lr 0.1 --epoch 30 --split balanced_scaffold --arch arch1 --resume ./resumes/videomol.pth --task_type classification --seed 0 --runseed 1 --image_aug_name simple
python finetune_video.py --dataroot ./datasets --dataset BTK --batch 32 --lr 0.1 --epoch 30 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type classification --seed 0 --runseed 0 --image_aug_name simple

python finetune_video.py --dataroot ./datasets --dataset CDK4-cyclinD3 --batch 32 --lr 0.05 --epoch 30 --split balanced_scaffold --arch arch1 --resume ./resumes/videomol.pth --task_type classification --seed 0 --runseed 2 --image_aug_name aug2
python finetune_video.py --dataroot ./datasets --dataset CDK4-cyclinD3 --batch 32 --lr 0.005 --epoch 30 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type classification --seed 0 --runseed 1 --image_aug_name aug2
python finetune_video.py --dataroot ./datasets --dataset CDK4-cyclinD3 --batch 32 --lr 0.05 --epoch 30 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type classification --seed 0 --runseed 0 --image_aug_name aug2

python finetune_video.py --dataroot ./datasets --dataset EGFR --batch 16 --lr 0.005 --epoch 30 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type classification --seed 0 --runseed 1 --image_aug_name simple
python finetune_video.py --dataroot ./datasets --dataset EGFR --batch 8 --lr 0.0001 --epoch 30 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type classification --seed 0 --runseed 0 --image_aug_name simple
python finetune_video.py --dataroot ./datasets --dataset EGFR --batch 8 --lr 0.001 --epoch 30 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type classification --seed 0 --runseed 2 --image_aug_name simple

python finetune_video.py --dataroot ./datasets --dataset FGFR1 --batch 32 --lr 0.03 --epoch 30 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type classification --seed 0 --runseed 0 --image_aug_name aug2
python finetune_video.py --dataroot ./datasets --dataset FGFR1 --batch 8 --lr 0.003 --epoch 30 --split balanced_scaffold --arch arch1 --resume ./resumes/videomol.pth --task_type classification --seed 0 --runseed 1 --image_aug_name aug2
python finetune_video.py --dataroot ./datasets --dataset FGFR1 --batch 8 --lr 0.001 --epoch 30 --split balanced_scaffold --arch arch1 --resume ./resumes/videomol.pth --task_type classification --seed 0 --runseed 2 --image_aug_name aug2

python finetune_video.py --dataroot ./datasets --dataset FGFR2 --batch 32 --lr 0.001 --epoch 10 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type classification --seed 0 --runseed 0 --image_aug_name aug2
python finetune_video.py --dataroot ./datasets --dataset FGFR2 --batch 32 --lr 0.003 --epoch 10 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type classification --seed 0 --runseed 2 --image_aug_name aug2
python finetune_video.py --dataroot ./datasets --dataset FGFR2 --batch 16 --lr 0.01 --epoch 30 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type classification --seed 0 --runseed 1 --image_aug_name aug2

python finetune_video.py --dataroot ./datasets --dataset FGFR3 --batch 32 --lr 0.1 --epoch 10 --split balanced_scaffold --arch arch1 --resume ./resumes/videomol.pth --task_type classification --seed 0 --runseed 2 --image_aug_name aug2
python finetune_video.py --dataroot ./datasets --dataset FGFR3 --batch 32 --lr 0.005 --epoch 30 --split balanced_scaffold --arch arch1 --resume ./resumes/videomol.pth --task_type classification --seed 0 --runseed 1 --image_aug_name aug2
python finetune_video.py --dataroot ./datasets --dataset FGFR3 --batch 8 --lr 0.0003 --epoch 10 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type classification --seed 0 --runseed 0 --image_aug_name aug2

python finetune_video.py --dataroot ./datasets --dataset FGFR4 --batch 16 --lr 0.008 --epoch 100 --split balanced_scaffold --arch arch1 --resume ./resumes/videomol.pth --task_type classification --seed 0 --runseed 1 --image_aug_name aug2
python finetune_video.py --dataroot ./datasets --dataset FGFR4 --batch 32 --lr 0.008 --epoch 100 --split balanced_scaffold --arch arch1 --resume ./resumes/videomol.pth --task_type classification --seed 0 --runseed 2 --image_aug_name aug2
python finetune_video.py --dataroot ./datasets --dataset FGFR4 --batch 8 --lr 0.1 --epoch 100 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type classification --seed 0 --runseed 0 --image_aug_name aug2

python finetune_video.py --dataroot ./datasets --dataset FLT3 --batch 8 --lr 0.0005 --epoch 30 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type classification --seed 0 --runseed 1 --image_aug_name simple
python finetune_video.py --dataroot ./datasets --dataset FLT3 --batch 8 --lr 0.0005 --epoch 30 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type classification --seed 0 --runseed 2 --image_aug_name simple
python finetune_video.py --dataroot ./datasets --dataset FLT3 --batch 8 --lr 0.005 --epoch 30 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type classification --seed 0 --runseed 0 --image_aug_name simple

python finetune_video.py --dataroot ./datasets --dataset KPCD3 --batch 16 --lr 0.03 --epoch 10 --split balanced_scaffold --arch arch1 --resume ./resumes/videomol.pth --task_type classification --seed 0 --runseed 1 --image_aug_name aug2
python finetune_video.py --dataroot ./datasets --dataset KPCD3 --batch 32 --lr 0.01 --epoch 10 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type classification --seed 0 --runseed 0 --image_aug_name aug2
python finetune_video.py --dataroot ./datasets --dataset KPCD3 --batch 8 --lr 0.005 --epoch 10 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type classification --seed 0 --runseed 2 --image_aug_name aug2

python finetune_video.py --dataroot ./datasets --dataset MET --batch 8 --lr 0.08 --epoch 50 --split balanced_scaffold --arch arch1 --resume ./resumes/videomol.pth --task_type classification --seed 0 --runseed 0 --image_aug_name aug2
python finetune_video.py --dataroot ./datasets --dataset MET --batch 16 --lr 0.08 --epoch 50 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type classification --seed 0 --runseed 1 --image_aug_name aug2
python finetune_video.py --dataroot ./datasets --dataset MET --batch 8 --lr 0.05 --epoch 50 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type classification --seed 0 --runseed 2 --image_aug_name aug2

