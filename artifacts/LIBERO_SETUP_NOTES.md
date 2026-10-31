# OpenVLA-OFT LIBERO 环境复现要点

## Base
- Ubuntu 22.04, A100-PCIE-40GB, 90GB RAM
- Miniconda Python 3.10.14
- PyTorch 2.2.0+cu121, torchvision 0.17.0, torchaudio 2.2.0
- numpy==1.26.4
- opencv-python==4.10.0.84
- tensorflow==2.15.0, tensorflow-metadata==1.14.0
- wandb==0.16.6
- transformers==4.40.1 (moojink/transformers-openvla-oft)
- LIBERO master tarball + openvla-oft editable install
- robosuite==1.4.1, mujoco==3.3.7
- CUDA toolkit/nvcc 12.1.105, flash-attn==2.5.5 (built for torch2.2/cu121)

## Key fixes
- LIBERO path: add /root/LIBERO to PYTHONPATH or a .pth file
- ~/.libero/config.yaml must point to LIBERO bddl_files/init_files/assets
- MuJoCo must be <=3.3.x for robosuite 1.4.1; 3.17 breaks get_joint_qpos_addr
- Do not install wandb 0.28 with protobuf 3.20; pin 0.16.6
- Do not install tensorflow-metadata 1.21 with TF2.15; pin 1.14.0
- OpenVLA checkpoint download through HF_ENDPOINT=https://hf-mirror.com
- Run with MUJOCO_GL=egl and PYOPENGL_PLATFORM=egl

## Evaluation
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint /root/models/openvla-oft-libero-spatial \
  --task_suite_name libero_spatial --num_trials_per_task 50

## Result
500 episodes, 493 successes, 98.6%.
