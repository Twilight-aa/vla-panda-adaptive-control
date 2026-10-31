# 远程 LIBERO-Spatial 500 回合基线结果

日期：2026-10-31。实例：Matpool，A100-PCIE-40GB，90GB RAM。

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

## 后续实验是否还需要 GPU

**当前 LIBERO-Spatial 基线已经完成，这个实例上暂时没有必须继续跑的实验；但整个课题后续仍然需要 GPU 服务器。**

后续需要 GPU 的工作包括：

1. 将自定义 CTC / SMC / RBF-SMC 力矩控制器接入 OpenVLA-OFT/LIBERO 的参考生成与执行链路。
2. 对完整系统做在线闭环评估，而不是只评估未修改的 OpenVLA-OFT。
3. 按正式实验矩阵做多方法、多工况、多种子统计；原计划核心矩阵约为 4 方法 × 6 工况 × 30 回合 = 720 回合，另有消融和失败案例。
4. 如论文需要，补跑 LIBERO-Object、Goal、Long 三个套件；每个套件 500 回合约 70–80 分钟。
5. 可能的小规模 LoRA 微调或动作头适配。

因此：

- 如果近期马上进入控制器接入/闭环实验，建议保留实例或制作快照，避免重装。
- 如果暂时不继续，当前数据已导出，可以清盘；下次用 `artifacts/LIBERO_SETUP_NOTES.md` 重新配置，约 30–60 分钟。
- 清盘不会影响已推送到 GitHub 的结果，但 15GB 检查点需要重新下载。

## 评测注意事项

- 本次只覆盖官方未修改的 OpenVLA-OFT 检查点，不代表接入自定义控制器后的性能。
- 本次只跑了 LIBERO-Spatial，不能外推到其他三个套件。
- 评测没有做多随机种子重复；500 回合是单个检查点/单次评测。
- 退出时的 EGL teardown 警告不影响 episode 结果。
