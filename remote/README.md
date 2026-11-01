# 远程 LIBERO 控制器接入

本目录是 2026-11-01 在 Matpool A100-PCIE-40GB 上实际运行过的代码和配置，用于复现 `docs/远程LIBERO控制器对比结果.md`。

## 文件

- `libero_torque_wrapper.py`：替换 robosuite OSC 的力矩律；保留 OSC 的笛卡尔目标，用阻尼最小二乘 IK 求 7 关节参考，以 20 Hz 计算 CTC / SMC / RBF-SMC。`install()` 必须包住 `env.reset()`：robosuite 每次 reset 都会重建 robot 和 OSC，只在创建后替换一次会被 reset 丢掉。
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

## 结果边界与接入 bug

2026-11-01 的 5×500 批次**无效**：robosuite 的 `env.reset()` 会重建 robot 和 `OperationalSpaceController`，旧 `install()` 只在环境创建后替换一次，reset 后包装器被丢掉；因此名义 CTC/SMC/RBF-SMC 实际都是 OSC，旧日志不能作为控制器结果。

修复后的 `install()` 已包住 `env.reset()`，诊断输出为：

```text
after reset: TorqueControllerWrapper
after step:  TorqueControllerWrapper, _steps=25, ||tau||=26.77
```

此外，`run_controller()` 在 500 Hz 子步调用，不能把 `(qtar-q)/dt`（按 20 Hz 取的 dt）当速度前馈；当前版本令 `dqd=0`，由 CTC/SMC 的 PD/滑模项跟踪 IK 阶跃目标。修复后的新实例 pilot（2026-11-01）：OSC 10/10；关节空间 IK CTC 4/10。改为任务空间 CTC（`cartesian=True`，关闭有限差分 `cart_use_jdot`）后，两次 10 回合 pilot 均为 9/10，均 task 9 失败；最终复核 8/10（task 8、9 失败）。pilot 仍未严格 10/10，因此没有启动 10×10×5 或 500 回合/方法。失败集中在高位/长轨迹任务，不是包装器没装上。当前包装器没有记录完整的轨迹级控制指标，下一轮应补关节/末端误差、IK 残差、力矩 RMS/饱和和接触力，再做扰动实验。完整审计见 `docs/远程LIBERO控制器对比结果.md`。
