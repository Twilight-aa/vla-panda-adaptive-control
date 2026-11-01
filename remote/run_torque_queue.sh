#!/bin/bash
while tmux has-session -t torque_ctc 2>/dev/null; do sleep 60; done
echo "CTC_FINISHED $(date)" >> /root/torque_queue.log
bash /root/run_torque_eval.sh SMC
echo "SMC_FINISHED $(date)" >> /root/torque_queue.log
bash /root/run_torque_eval.sh RBF_SMC
echo "RBF_FINISHED $(date)" >> /root/torque_queue.log
