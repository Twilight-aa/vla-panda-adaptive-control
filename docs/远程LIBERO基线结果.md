# 远程 LIBERO-Spatial 500 回合基线结果

日期：2026-10-31。实例：Matpool，A100-PCIE-40GB，90GB RAM。

> **2026-11-01 08:50 更正：** 原计划比较的 5 个自定义力矩控制器批次无效。事后诊断发现 robosuite 的 `env.reset()` 会重建 robot 和 OSC 控制器，旧 `install()` 只在环境创建后替换一次，reset 后包装器被丢掉；因此 `/root/eval_torque_*_t50.log` 的 5×500 回合实际执行的是 OSC，不能作为控制器结果。修复后的代码已验证 `env.reset()` 后仍是 `TorqueControllerWrapper`，但正确的正式批次尚未完成。详见 `docs/远程LIBERO控制器对比结果.md`。**本文件下面的 500 回合 OSC 基线（493/500）仍然有效。**

## 结果

- 检查点：`moojink/openvla-7b-oft-finetuned-libero-spatial`
- 任务套件：`libero_spatial`
- 10 tasks × 50 trials = **500 episodes**
- 成功：**493/500**
- 成功率：**98.6%**
- 失败：7 个回合，集中在 task 4、5、7、9

|任务|成功/50|成功率|
|---|---:|---:|
|0 碗在盘子和调料架之间|50/50|100%|
|1 碗在调料架旁|50/50|100%|
|2 碗在桌子中央|50/50|100%|
|3 碗在饼干盒上|50/50|100%|
|4 碗在木柜上层抽屉|48/50|96%|
|5 碗在调料架上|47/50|94%|
|6 碗在饼干盒旁|50/50|100%|
|7 碗在炉灶上|49/50|98%|
|8 碗在盘子旁|50/50|100%|
|9 碗在木柜上|49/50|98%|

结果与 OpenVLA-OFT 论文报告的 LIBERO-Spatial 约 97.6% 基本一致。早期 50/50 和 200/200 的 100% 是小样本偏乐观，不能当作完整基准结论。

## 运行环境

- Ubuntu 22.04，Python 3.10.14
- PyTorch 2.2.0+cu121
- CUDA 12.1，flash-attn 2.5.5
- transformers 4.40.1（OpenVLA-OFT fork）
- LIBERO + robosuite 1.4.1 + MuJoCo 3.3.7
- TensorFlow 2.15.0，numpy 1.26.4
- `tensorflow-metadata==1.14.0`、`wandb==0.16.6`、`opencv-python==4.10.0.84` 用于解决依赖冲突

## 可复现产物

- `artifacts/libero_spatial_500_export.tar.gz`：36MB，包含完整日志、逐任务结果、rollout 视频、pip freeze、setup notes、LIBERO 配置和 OpenVLA-OFT commit。
- `artifacts/libero_spatial_500_summary.json`：逐任务成功率。
- `artifacts/LIBERO_SETUP_NOTES.md`：环境重装要点。
- SHA-256：`3f9020364a11bb89dabe0547ad71d9e4dc51128f889014c08f657b3db998161c`
- 自定义控制器批次的归档（**已标记无效，实际为 OSC**）另见 `artifacts/torque_500_logs_export.tar.gz`（210KB，日志/脚本/源码）、`artifacts/torque_500_summary.json` 和 `artifacts/torque_500_rollouts_2026_11_01.tar.gz`（121MB，2550 个视频）；原因见 `docs/远程LIBERO控制器对比结果.md`。

## 后续实验是否还需要 GPU

**当前计划内的 OSC 基线（500 回合）已完成；自定义控制器批次因接入 bug 无效，修复后尚未重跑。实例上的 GPU 任务目前暂停，等待修复验证和正式批次决定；整个课题后续仍然需要 GPU 服务器。**

后续需要 GPU 的工作包括：

1. 用修复后的包装器重跑自定义控制器批次，先做小规模配对 pilot，再决定是否扩大到 500 回合/方法。
2. 给自定义控制器补轨迹级遥测：关节/末端跟踪误差、IK 残差、力矩 RMS 与饱和比例、接触力、夹爪滑移。
3. 在负载、摩擦、动作噪声、观测噪声和执行延迟等扰动工况下做在线闭环评估；LIBERO-Spatial 的成功率已经饱和，不适合区分控制器。
4. 如论文需要，补跑 LIBERO-Object、Goal、Long 三个套件；每个套件 500 回合约 70–80 分钟。
5. 可能的小规模 LoRA 微调或动作头适配。

因此：

- 如果马上继续修复验证/正式批次，建议保留实例或制作快照，避免重装。
- 如果暂时不继续，当前 OSC 基线数据已导出，可以清盘；下次用 `artifacts/LIBERO_SETUP_NOTES.md` 重新配置，约 30–60 分钟。
- 清盘不会影响已导出的 OSC 基线，但 15GB 检查点需要重新下载。

## 评测注意事项

- 本文件的总表是**原始 OSC 基线**，仍有效；名义 CTC/SMC/RBF-SMC 的 5×500 批次实际也是 OSC，已标记无效，见 `docs/远程LIBERO控制器对比结果.md`。
- OSC 基线只覆盖 LIBERO-Spatial、一个检查点、一个评测种子；500 回合是评测回合数，不是 500 个独立训练种子。
- 退出时的 EGL teardown 警告不影响 episode 结果。
