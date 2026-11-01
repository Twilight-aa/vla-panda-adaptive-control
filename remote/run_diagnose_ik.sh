#!/usr/bin/env bash
set -o pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate openvla-oft
cd /root/openvla-oft/experiments/robot/libero
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH=/root/LIBERO:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false
python diagnose_ik.py > /root/diagnose_ik.log 2>&1
rc=$?
echo "EXIT:$rc" >> /root/diagnose_ik.log
exit $rc
