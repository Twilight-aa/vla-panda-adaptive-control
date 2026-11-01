# 远程 LIBERO-Spatial 控制器接入审计

更新时间：2026-11-01 08:50。实例：Matpool，A100-PCIE-40GB。

## 0. 当前结论

**原计划用于比较 CTC/SMC/RBF-SMC 的 5×500 回合批次无效：实际执行的是 robosuite 原始 OSC，不是自定义力矩控制器。** 事后诊断发现，robosuite 的 `env.reset()` 会重建 robot 和它的 `OperationalSpaceController`；原 `install()` 只在环境创建后替换一次 `env.robots[0].controller`，第一次 `env.reset()` 就把包装器丢掉了。因此之前记录的 98.2%–99.0% 只是 OSC 的重复评测，不能作为控制器对比证据。

诊断证据（同一环境）：

```text
before install: OperationalSpaceController
after install:  TorqueControllerWrapper
after env.reset: OperationalSpaceController   # 原实现，包装器被丢掉
```

修复后重新验证：

```text
after reset: TorqueControllerWrapper
after step:  TorqueControllerWrapper, _steps=25, ||tau||=26.77
```

**正确批次尚未完成。** 旧数字、旧报告结论和基于它们的“RBF-SMC 与 OSC 无差异”说法全部撤回；不能用这批数据支持或反驳任何控制器假设。原始 OSC 基线 493/500（98.6%）本身仍然有效，因为它就是 OSC。

## 1. 原批次记录（无效）

这批日志仍保留用于审计，但不能当作五个控制器的结果。

|名义方法|成功/500|实际控制器|状态|
|---|---:|---|---|
|OSC 基线|493|OSC|有效基线|
|名义 CTC|493|OSC（reset 后）|无效|
|名义 SMC|495|OSC（reset 后）|无效|
|名义 RBF-SMC|492|OSC（reset 后）|无效|
|名义 RBF-SMC 无鲁棒项|491|OSC（reset 后）|无效|
|名义 RBF-SMC 冻结|491|OSC（reset 后）|无效|

五组名义方法的差异只有 1–4 个回合，和“它们其实是同一个 OSC 控制器”完全一致；这也说明 OSC/VLA 自身存在约 1 个百分点的 run-to-run 波动。它不能用来证明控制器等价，更不能用来证明某个控制器更优。

## 2. 根因

`run_libero_eval.py` 的每个 episode 都从 `env.reset()` 开始：

```python
# experiments/robot/libero/run_libero_eval.py, run_episode()
env.reset()
obs = env.set_init_state(initial_state)
```

robosuite 的 reset 链路会重新加载控制器：

```text
OffScreenRenderEnv.reset()
  -> robot.reset()
     -> Robot.reset()
        -> self._load_controller()
           -> self.controller = controller_factory(OSC_POSE, ...)
```

原实现只在 `get_libero_env()` 返回后做一次：

```python
env.robots[0].controller = TorqueControllerWrapper(env.robots[0].controller, ...)
```

第一次 `env.reset()` 后 `env.robots[0]` 甚至是一个新的 robot 实例（实测 `env.robots[0] is old_robot == False`），所以旧 robot 上的补丁也不会生效。此后所有控制步都由新的 OSC 执行。

## 3. 修复

`remote/libero_torque_wrapper.py` 的 `install()` 现在包住 `env.reset`：

```python
original_reset = env.reset

def reset_and_wrap(*args, **kwargs):
    result = original_reset(*args, **kwargs)
    robot = env.robots[0]
    robot.controller = TorqueControllerWrapper(robot.controller, method, config)
    return result

env.reset = reset_and_wrap
```

同时做了两项性能修复，否则 500 回合不可行：

- IK 目标按笛卡尔 goal 缓存，每个策略步只解一次，而不是每个 20 Hz 控制子步都解 80 次；
- IK 复用同一个 `MjData`，迭代上限从 80 降到 40，并在误差小于 `1e-4` 时提前退出。

修复后在无模型推理的空动作测试中，`env.step` 约 0.08–0.1 s；修复前约 0.5 s/步（旧 pilot 约 2 分钟/回合）。这仍比 OSC 慢，完整 5×500 需要重新评估时间预算。

2026-11-01 08:53 又修正了一处控制律问题：初版包装器把 `(qtar - q) / dt` 作为速度前馈，但 `run_controller()` 实际以 500 Hz 的子步调用，而 `dt` 按 20 Hz 写成 0.05 s；该前馈在仿真子步上过大，容易把关节推向饱和。修复后 `dqd=0`，由 CTC/SMC 的 PD/滑模项直接跟踪 IK 阶跃目标。初版修复的 pilot 在任务 0 连续 3 次失败后已停止；改用 `dqd=0` 的 v2 pilot 第一个 episode 仍失败（08:55 前，日志 `/root/fixed_CTC_t1_v2.log`），说明问题不只是速度前馈。

调试 telemetry（`TORQUE_DEBUG=1`）显示自定义控制器确实在执行，但 CTC 输出严重饱和：

```text
[torque-debug] step=1   q_err=0.0370 ee_err=0.0000 |tau|=184.99 max|tau|=151.46
[torque-debug] step=501 q_err=0.3001 ee_err=0.0532 |tau|=236.35 max|tau|=174.96
```

随后加入一阶参考滤波（`ref_tau=0.1`，默认；可配置），非 VLA 的空动作跟踪测试里关节误差降到约 `1e-4`、力矩峰值降到约 50（未饱和），但末端仍有约 2.5 cm 的稳态误差；`_ik` 在移动目标上记录了约 2.5 cm 的位置残差（`|do|` 很小），说明 IK 在连续移动目标/接近奇异位形时需要进一步检查迭代、阻尼和关节限位，而不是简单加大增益。另一个 `ref_tau=0.02` 的测试仍未消除这 2.5 cm 残差，问题更像 IK 收敛/可达性而不是滤波器。

远程实例在 2026-11-01 09:00 左右的 IK 收敛测试中失联：TCP 端口仍可连接，但 SSH 在密钥交换阶段被远端关闭（`kex_exchange_identification: Connection closed by remote host`）。修复后的 pilot 因此中断；在恢复实例前无法继续确认 IK 残差或跑正式批次。

## 4. 正确批次的当前状态

- 修复后的 CTC 10 回合 pilot（10 tasks × 1 trial）没有通过：`fixed_CTC_t1_v2.log` 第一个 episode 仍然失败；调试确认控制器在跑（`_steps`/`tau` 非零），但 CTC 力矩饱和，随后又暴露出移动目标上的 IK 位置残差约 2.5 cm。初版 `fixed_CTC_t1.log` 在任务 0 连续 3 次失败后已停止，不作为结果。
- 一阶参考滤波和 `ref_tau` 调整只降低了力矩峰值/关节跟踪误差，没有消除 IK 的 2.5 cm 残差；下一步应检查 IK 的可达性、奇异位形、阻尼和关节限位，而不是继续调增益。
- 2026-11-01 09:00 左右远程实例失联；pilot 中断，无法继续调试或重跑。当前没有可报告的自定义控制器成功率。
- 修复后尚未重跑 5×500 正式批次；旧 `/root/eval_torque_*_t50.log`、`/root/torque_core_summary.json` 和 `artifacts/torque_500_*` 仍保留，但已标记为无效/OSC 重复；不要在新报告或论文中引用其“方法差异”。
- 原有的 OSC 500 回合基线（`/root/eval_spatial_t50.log`，493/500）有效，继续作为基线。

## 5. 影响与下一步

1. **撤回旧结论。** 旧报告中“CTC/SMC/RBF-SMC 与 OSC 相当”“RBF-SMC 未显示收益”等表述全部不成立；那批数据没有接入自定义控制器。
2. **先验证修复后的 10 回合 pilot。** 如果 pilot 正常，再决定正式矩阵规模；不能再把一次 500 回合当成“已经完成”。
3. **正式批次加轨迹级遥测。** 记录关节/末端跟踪误差、IK 残差、力矩 RMS/饱和、接触力和夹爪滑移；只测成功/失败无法区分控制器，而且当前任务成功率本来就会饱和。
4. **重估时间预算。** 修复后自定义控制器的单步开销高于 OSC；5 个方法 ×500 回合可能超过 6 小时。可选方案是先在 10 任务 ×10–20 回合 ×5 方法上做配对遥测，再对最有希望的方法扩大到 500 回合。
5. **保留 OSC 基线。** 493/500 是有效数据；但不要把它和无效的“自定义方法”表混在一起。

## 6. 产物

有效：

- `artifacts/libero_spatial_500_export.tar.gz`：原始 OSC 500 回合基线（36MB，含视频、日志、环境说明）。
- `docs/远程LIBERO基线结果.md`：有效基线结果。

无效/仅审计：

- `/root/eval_torque_{CTC,SMC,RBF_SMC,RBF_NO_ROBUST,RBF_FROZEN}_t50.log`：名义控制器日志，实际为 OSC。
- `artifacts/torque_500_logs_export.tar.gz`：上述无效日志的归档；JSON 已加 `valid=false`。
- `artifacts/torque_500_rollouts_2026_11_01.tar.gz`：对应 2550 个视频，实际也是 OSC；仅作审计/负对照。
- `artifacts/torque_500_summary.json`、`artifacts/torque_500_pairwise_stats.json`：已标记无效，不再作为控制器结果。

修复代码：

- `remote/libero_torque_wrapper.py`：包住 `env.reset`，并缓存 IK。
- `remote/libero_torque_eval.py`：评测入口不变。
- `remote/README.md`：远程复现说明；其中“结果边界”已改写。

## 7. 复现诊断

```bash
cd /root/openvla-oft/experiments/robot/libero
python - <<'PY'
import run_libero_eval as rle
from libero.libero import benchmark
from libero_torque_wrapper import install
suite = benchmark.get_benchmark_dict()["libero_spatial"]()
task = suite.get_task(0)
env, _ = rle.get_libero_env(task, "openvla", 256)
print("before:", type(env.robots[0].controller).__name__)
install(env, "CTC", {"ctc_kp": 900., "ctc_kd": 66.})
env.reset()
print("after reset:", type(env.robots[0].controller).__name__)
PY
```

期望输出 `after reset: TorqueControllerWrapper`；若仍看到 `OperationalSpaceController`，说明安装仍被 reset 覆盖，不能开始正式批次。
