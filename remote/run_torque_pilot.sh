#!/bin/bash
set -o pipefail
method=$1
source /root/miniconda3/etc/profile.d/conda.sh
conda activate openvla-oft
cd /root/openvla-oft
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH=/root/LIBERO:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
python experiments/robot/libero/libero_torque_eval.py --method "$method" --trials 1 > "/root/pilot_${method}.log" 2>&1
rc=$?
echo "EXIT:$rc" >> "/root/pilot_${method}.log"
exit $rc
