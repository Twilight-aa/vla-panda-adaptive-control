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

## 2026-11-01 新实例重建（px-cloud2）

旧实例失联后，新实例 `px-cloud2.matpool.com:28394` 重新配置：

- Python 3.10.14、torch 2.2.0+cu121、torchvision 0.17.0+cu121、torchaudio 2.2.0+cu121
- numpy 1.26.4、opencv-python 4.10.0.84、tensorflow 2.15.0、tensorflow-metadata 1.14.0、wandb 0.16.6、protobuf 3.20.3
- robosuite 1.4.1、mujoco 3.3.7、LIBERO 0.1.0
- 检查点从 hf-mirror 下载到 `/root/models/openvla-oft-libero-spatial`
- `flash-attn==2.5.5` 的预编译 wheel 下载超时，源码构建未完成；当前模型以 eager attention 运行，OSC 10/10 和 CTC 4/10 pilot 均可正常推理。若后续训练/需要 flash attention，需用 `FLASH_ATTENTION_FORCE_BUILD=TRUE` 从源码构建或换可访问的 wheel 源。

启动 pilot 的关键环境变量：

```bash
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYTHONPATH=/root/LIBERO:${PYTHONPATH:-}
source /root/miniconda3/etc/profile.d/conda.sh
conda activate openvla-oft
```
