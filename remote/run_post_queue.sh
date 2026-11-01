#!/bin/bash
while tmux has-session -t torque_queue 2>/dev/null; do sleep 60; done
echo "CORE_QUEUE_DONE $(date)" >> /root/post_queue.log
source /root/miniconda3/etc/profile.d/conda.sh
conda activate openvla-oft
python /root/summarize_torque_results.py >> /root/post_queue.log 2>&1
echo "--- pilots ---" >> /root/post_queue.log
bash /root/run_torque_pilot.sh RBF_NO_ROBUST
bash /root/run_torque_pilot.sh RBF_FROZEN
python /root/check_pilots.py >> /root/post_queue.log 2>&1
if [ $? -ne 0 ]; then echo "ABORT_AFTER_PILOT $(date)" >> /root/post_queue.log; exit 1; fi
echo "--- formal ablations ---" >> /root/post_queue.log
bash /root/run_torque_eval.sh RBF_NO_ROBUST
bash /root/run_torque_eval.sh RBF_FROZEN
echo "ALL_ABLATIONS_DONE $(date)" >> /root/post_queue.log
