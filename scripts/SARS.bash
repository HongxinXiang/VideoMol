source activate videomol
cd your_command_path

python finetune-video.py --dataroot ./datasets --dataset 3CL --batch 32 --lr 0.003 --epoch 20 --split balanced_scaffold --arch arch1 --resume ./resumes/videomol.pth --task_type classification --seed 0 --runseed 0
python finetune-video.py --dataroot ./datasets --dataset 3CL --batch 64 --lr 0.003 --epoch 30 --split balanced_scaffold --arch arch1 --resume ./resumes/videomol.pth --task_type classification --seed 0 --runseed 1 
python finetune-video.py --dataroot ./datasets --dataset 3CL --batch 8 --lr 0.0005 --epoch 10 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type classification --seed 0 --runseed 2

python finetune-video.py --dataroot ./datasets --dataset ACE2 --batch 8 --lr 1e-05 --epoch 60 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type classification --seed 0 --runseed 1 
python finetune-video.py --dataroot ./datasets --dataset ACE2 --batch 64 --lr 0.003 --epoch 30 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type classification --seed 0 --runseed 2 
python finetune-video.py --dataroot ./datasets --dataset ACE2 --batch 8 --lr 0.0008 --epoch 60 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type classification --seed 0 --runseed 0

python finetune-video.py --dataroot ./datasets --dataset hCYTOX --batch 8 --lr 0.0005 --epoch 40 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type classification --seed 0 --runseed 2 
python finetune-video.py --dataroot ./datasets --dataset hCYTOX --batch 16 --lr 1e-05 --epoch 10 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type classification --seed 0 --runseed 1 
python finetune-video.py --dataroot ./datasets --dataset hCYTOX --batch 8 --lr 0.0005 --epoch 30 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type classification --seed 0 --runseed 0

python finetune-video.py --dataroot ./datasets --dataset MERS-PPE_cs --batch 8 --lr 0.0005 --epoch 20 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type classification --seed 0 --runseed 2 
python finetune-video.py --dataroot ./datasets --dataset MERS-PPE_cs --batch 8 --lr 5e-06 --epoch 10 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type classification --seed 0 --runseed 1 
python finetune-video.py --dataroot ./datasets --dataset MERS-PPE_cs --batch 8 --lr 0.0005 --epoch 20 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type classification --seed 0 --runseed 0

python finetune-video.py --dataroot ./datasets --dataset MERS-PPE --batch 32 --lr 0.01 --epoch 10 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type classification --seed 0 --runseed 2 
python finetune-video.py --dataroot ./datasets --dataset MERS-PPE --batch 8 --lr 0.01 --epoch 10 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type classification --seed 0 --runseed 1 
python finetune-video.py --dataroot ./datasets --dataset MERS-PPE --batch 16 --lr 0.01 --epoch 10 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type classification --seed 0 --runseed 0

python finetune-video.py --dataroot ./datasets --dataset CoV1-PPE_cs --batch 32 --lr 0.01 --epoch 60 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type classification --seed 0 --runseed 0 
python finetune-video.py --dataroot ./datasets --dataset CoV1-PPE_cs --batch 128 --lr 0.005 --epoch 30 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type classification --seed 0 --runseed 1 
python finetune-video.py --dataroot ./datasets --dataset CoV1-PPE_cs --batch 128 --lr 0.008 --epoch 30 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type classification --seed 0 --runseed 2

python finetune-video.py --dataroot ./datasets --dataset CoV1-PPE --batch 32 --lr 0.0005 --epoch 20 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type classification --seed 0 --runseed 2 
python finetune-video.py --dataroot ./datasets --dataset CoV1-PPE --batch 16 --lr 0.0005 --epoch 10 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type classification --seed 0 --runseed 1 
python finetune-video.py --dataroot ./datasets --dataset CoV1-PPE --batch 8 --lr 0.0005 --epoch 10 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type classification --seed 0 --runseed 0

python finetune-video.py --dataroot ./datasets --dataset CPE --batch 8 --lr 0.0005 --epoch 30 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type classification --seed 0 --runseed 1 
python finetune-video.py --dataroot ./datasets --dataset CPE --batch 8 --lr 0.0005 --epoch 10 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type classification --seed 0 --runseed 0 
python finetune-video.py --dataroot ./datasets --dataset CPE --batch 8 --lr 5e-06 --epoch 10 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type classification --seed 0 --runseed 2

python finetune-video.py --dataroot ./datasets --dataset cytotox --batch 16 --lr 0.0005 --epoch 20 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type classification --seed 0 --runseed 0 
python finetune-video.py --dataroot ./datasets --dataset cytotox --batch 32 --lr 0.0005 --epoch 20 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type classification --seed 0 --runseed 2 
python finetune-video.py --dataroot ./datasets --dataset cytotox --batch 32 --lr 0.0005 --epoch 20 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type classification --seed 0 --runseed 1

python finetune-video.py --dataroot ./datasets --dataset AlphaLISA --batch 16 --lr 1e-05 --epoch 10 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type classification --seed 0 --runseed 1 
python finetune-video.py --dataroot ./datasets --dataset AlphaLISA --batch 8 --lr 5e-06 --epoch 10 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type classification --seed 0 --runseed 0 
python finetune-video.py --dataroot ./datasets --dataset AlphaLISA --batch 16 --lr 0.0005 --epoch 30 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type classification --seed 0 --runseed 2

python finetune-video.py --dataroot ./datasets --dataset TruHit --batch 8 --lr 5e-06 --epoch 20 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type classification --seed 0 --runseed 0 
python finetune-video.py --dataroot ./datasets --dataset TruHit --batch 32 --lr 5e-05 --epoch 10 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type classification --seed 0 --runseed 1 
python finetune-video.py --dataroot ./datasets --dataset TruHit --batch 8 --lr 5e-06 --epoch 20 --split balanced_scaffold --arch arch3 --resume ./resumes/videomol.pth --task_type classification --seed 0 --runseed 2 
