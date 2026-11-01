# 远程 LIBERO 控制器接入

本目录是 2026-11-01 在 Matpool A100-PCIE-40GB 上实际运行过的代码和配置，用于复现 `docs/远程LIBERO控制器对比结果.md`。

## 文件

- `libero_torque_wrapper.py`：替换 robosuite OSC 的力矩律；保留 OSC 的笛卡尔目标，用阻尼最小二乘 IK 求 7 关节参考，以 20 Hz 计算 CTC / SMC / RBF-SMC。
- `libero_torque_eval.py`：加载 OpenVLA-OFT/LIBERO 评测入口；在每个环境创建后安装包装器。
- `selected.json`：本机冻结后用于远程实验的控制器参数；`RBF_NO_ROBUST` 和 `RBF_FROZEN` 沿用 `RBF_SMC` 参数。
- `run_torque_eval.sh`、`run_torque_queue.sh`、`run_post_queue.sh`、`run_torque_pilot.sh`：远程队列脚本，保留原始路径供核对。
- `summarize_torque_results.py`：逐行解析评测日志，输出总成功率、逐任务成功率和 Wilson 区间；旧版单条大正则会被 rich/ANSI 换行破坏，已重写。
- `check_pilots.py`：检查 pilot 日志中的最终结果。

## 远程环境

以下路径来自实际实例，重建时需按新实例调整：

```text
/root/openvla-oft/experiments/robot/libero/   # libero_torque_*.py
/root/models/openvla-oft-libero-spatial/      # 15 GB 检查点
/root/eval_torque_<METHOD>_t50.log
/root/selected.json
/root/torque_core_summary.json
```

环境固定为 Python 3.10.14、PyTorch 2.2.0+cu121、CUDA 12.1、robosuite 1.4.1、MuJoCo 3.3.7、flash-attn 2.5.5、transformers 4.40.1（OpenVLA-OFT fork）。完整依赖见 `artifacts/LIBERO_SETUP_NOTES.md`。

## 复现

```bash
cd /root/openvla-oft/experiments/robot/libero
python libero_torque_eval.py --method CTC --trials 50
python libero_torque_eval.py --method SMC --trials 50
python libero_torque_eval.py --method RBF_SMC --trials 50
python libero_torque_eval.py --method RBF_NO_ROBUST --trials 50
python libero_torque_eval.py --method RBF_FROZEN --trials 50
```

评测日志汇总：

```bash
python remote/summarize_torque_results.py \
  --log-dir /root \
  --baseline /root/eval_spatial_t50.log \
  --out /root/torque_core_summary.json
```

## 结果边界

各方法 500 回合的成功率为 98.2%–99.0%，与 OSC 基线 98.6% 没有统计显著差异；任务成功率接近天花板，不能据此宣称 RBF-SMC 更优。当前包装器没有记录轨迹级控制指标，下一轮应先补关节/末端误差、IK 残差、力矩 RMS/饱和和接触力，再做扰动实验。
