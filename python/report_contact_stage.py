"""Regenerate the contact/fair-tuning report from the clean v2 data.

Adds the contact-audit resolution, the clean rerun comparison, and the
independent gripper capacity envelope.  No simulation is run here.
"""
import csv, json, hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_panda_local import ROOT

FT = ROOT / 'results/fair_tuning_v2'
OLD = ROOT / 'results/fair_tuning_v1_original'
CT = ROOT / 'results/grasp_transport'
GB = ROOT / 'results/gripper_bench'
CA = ROOT / 'results/contact_audit'
CT_EXT = ROOT / 'results/grasp_transport_ext'
CT_15 = ROOT / 'results/grasp_transport_grip15'
METHODS = ['CTC', 'SMC', 'RBF_SMC']


def load(path):
    return list(csv.DictReader(path.open()))


def means(rows, key, scale=1.):
    return float(np.mean([float(r[key]) * scale for r in rows]))


dev = load(FT / 'development.csv')
held = load(FT / 'heldout.csv')
noise = load(FT / 'noise_test.csv')
grasp = load(CT / 'summary.csv')
grasp90 = load(CT_EXT / 'summary.csv')
grasp15 = load(CT_15 / 'summary.csv')
cap = load(GB / 'pull_sweep.csv')
old_dev = load(OLD / 'development.csv')
old_held = load(OLD / 'heldout.csv')
old_noise = load(OLD / 'noise_test.csv')
assert len(dev) == 96 and len(held) == 126 and len(noise) == 27 and len(grasp) == 45
assert len(grasp90) == 90 and len(grasp15) == 90
assert len(old_dev) == 96 and len(old_held) == 126 and len(old_noise) == 27
assert len(cap) == 20

selected = json.loads((FT / 'selected.json').read_text())
old_selected = json.loads((OLD / 'selected.json').read_text())
selected_identical = selected == old_selected
# matching rows by all identifying keys
keys = ['candidate', 'scenario', 'controller', 'seed', 'trajectory', 'noise_std']
max_ee_diff = 0.
for new_rows, old_rows in [(dev, old_dev), (held, old_held), (noise, old_noise)]:
    def key(r):
        return tuple(r[k] for k in keys if k in r)
    new_map = {key(r): r for r in new_rows}
    old_map = {key(r): r for r in old_rows}
    assert new_map.keys() == old_map.keys()
    for k in new_map:
        max_ee_diff = max(max_ee_diff, abs(float(new_map[k]['ee_rmse_m']) - float(old_map[k]['ee_rmse_m'])))

pre = json.loads((CA / 'payload_s23_t0_CTC.json').read_text())
pre_contacts = pre['contacts']
pre_pairs = sorted({tuple(r['pair']) for r in pre_contacts})
pre_max_force = max((r['force_norm'] for r in pre_contacts), default=0.)
fixed = json.loads((CA / 'payload_s23_t0_CTC_fixed.json').read_text())

# Capacity summary by friction.
cap_by_fric = {}
for fric in sorted({float(r['friction']) for r in cap}, reverse=True):
    vals = [float(r['downward_capacity_N']) for r in cap if abs(float(r['friction']) - fric) < 1e-12]
    cap_by_fric[fric] = (float(np.mean(vals)), float(np.min(vals)), float(np.max(vals)))
weights = {0.1: .1 * 9.81, 0.25: .25 * 9.81}

lines = [
    '# 等预算调参与物理抓取阶段报告',
    '',
    '本报告由 `python/report_contact_stage.py` 从 `results/fair_tuning_v2`、`results/grasp_transport` 和 `results/gripper_bench` 生成。2026-10-31 完成接触审计后重新生成本文件，旧数据保留在 `results/fair_tuning_v1_original` 与 `results/grasp_transport_v1_original`。',
    '',
    '## 关键结论',
    '',
    '- 接触审计已闭环：旧独立测试和噪声测试中的非零接触全部是负载工况初始化时夹爪指尖垫的零力单步接触，不是地面、手臂或环境碰撞。修复后 249 个自由运动回合全部零接触，冻结参数与指标复现。',
    '- 充分调参后，CTC 在当前独立轨迹测试的平均跟踪误差上优于所选 SMC 与 RBF-SMC。RBF-SMC 相比 SMC 在负载类工况仍有改善，但必须同时评价控制努力、噪声和接触结果。',
    '- 论文正式接触实验平台采用 1.5× 校准夹爪：5 工况 × 3 方法 × 6 种子共 90 回合，总体 72/90；名义、重物、0.35 摩擦、脉冲均 18/18，0.12 摩擦 0/18 作为失败边界。',
    '- 1× 原始 Menagerie 夹爪保留为敏感性/失败边界：标称和脉冲 3/3，重物、0.35、0.12 摩擦均 0/3。夹爪上拉试验表明这是夹持摩擦容量限制，不是手臂跟踪失败。',
    '',
    '- CTC 的增益位于本次候选范围高端，有限预算搜索不等于全局最优；测试结果没有用于重新选参数。',
    '',
    '## 接触审计：原因、修复与复核',
    '',
    '旧版 `setup(payload)` 在给手部增加负载质量和惯量后调用 `mj_setConst`。该调用重置了 `MjData` 状态（`qpos` 全零），随后 `run()` 只恢复了 7 个手臂关节，夹爪关节停留在 0（闭合），而不是 home keyframe 的 0.04（张开）。第一步 `mj_step` 时左右指尖垫恰好相切，产生接近零距离、零力的接触；夹爪指令 255 随后把手指打开，接触在一步后消失。',
    '',
    f'- 旧异常接触对：`{pre_pairs[0][0]} x {pre_pairs[0][1]}` 等指尖垫组合；全部出现在 `t=0`，记录到的接触力均为 0 N，穿透约 1e-17 m。',
    f'- 旧数据受影响范围：独立测试 24/126 回合（全部 seed=23 且为含负载工况），噪声测试 9/27 回合（全部 seed=43 且为 combined）；开发集 0/96。',
    '- 修复：在 `mj_setConst` 之后重新应用 home keyframe；核心 `run()` 增加 `initial_contacts`、`contact_steps`、`contact_pairs`、`max_contact_force_N`、`first_contact_time_s` 日志。',
    f'- 复核：修复后同一异常种子复现为 {fixed["summary"]["n_contact_steps"]} 个接触步；独立测试 126/126、噪声 27/27、开发 96/96 均为零接触。',
    f'- 结果复现：`selected.json` 与旧版完全相同（{selected_identical}）；249 个回合末端 RMSE 最大差异 {max_ee_diff:.2e} m。',
    '',
    '|数据集|旧版接触回合|修复后接触回合|',
    '|---|---:|---:|',
    f'|开发（96）|{sum(float(r["max_contacts"]) > 0 for r in old_dev)}|{sum(int(r["contact_steps"]) > 0 for r in dev)}|',
    f'|独立测试（126）|{sum(float(r["max_contacts"]) > 0 for r in old_held)}|{sum(int(r["contact_steps"]) > 0 for r in held)}|',
    f'|噪声测试（27）|{sum(float(r["max_contacts"]) > 0 for r in old_noise)}|{sum(int(r["contact_steps"]) > 0 for r in noise)}|',
    '',
    '## 等预算调参与独立测试',
    '',
    '开发集每方法 8 个候选，每候选 2 个开发种子 × 2 工况，共 96 回合；独立测试为 2 条参考 × seeds=21,22,23 × 7 工况 × 3 方法，共 126 回合。开发评分固定为 `末端 RMSE(mm) + 0.1 × 力矩 RMS(Nm) + 100 × 饱和比例`。',
    '',
    '选中参数：',
    '',
    '```json',
    json.dumps(selected, indent=2, ensure_ascii=False),
    '```',
    '',
    '独立测试均值，单位 mm 与 Nm：',
    '',
    '|工况|CTC RMSE|SMC RMSE|RBF-SMC RMSE|CTC 力矩 RMS|SMC 力矩 RMS|RBF-SMC 力矩 RMS|',
    '|---|---:|---:|---:|---:|---:|---:|',
]
scenarios = list(dict.fromkeys(r['scenario'] for r in held))
for sc in scenarios:
    vals = []
    for key, scale in [('ee_rmse_m', 1000), ('torque_rms_Nm', 1)]:
        vals.extend(means([r for r in held if r['scenario'] == sc and r['controller'] == m], key, scale)
                    for m in METHODS)
    lines.append('|' + sc + '|' + '|'.join(f'{v:.3f}' for v in vals) + '|')
lines += [
    '',
    'RBF-SMC 相对 SMC 在 payload、combined、unseen_payload、low_torque 上的平均 RMSE 更低；但 CTC 在所有列出的工况上误差最低。不能把某一方法在单一工况的领先推广为普遍优势。',
    '',
    '## 观测噪声检查',
    '',
    '控制器读取带噪关节角与速度，指标使用真实仿真状态；每格三种子，不重新调参。',
    '',
    '|噪声标准差|CTC RMSE(mm)|SMC RMSE(mm)|RBF-SMC RMSE(mm)|CTC 力矩 RMS(Nm)|',
    '|---|---:|---:|---:|---:|',
]
for std in [0, .001, .005]:
    vals = [means([r for r in noise if float(r['noise_std']) == std and r['controller'] == m], 'ee_rmse_m', 1000) for m in METHODS]
    ctc_torque = means([r for r in noise if float(r['noise_std']) == std and r['controller'] == 'CTC'], 'torque_rms_Nm')
    lines.append(f'|{std}|' + '|'.join(f'{v:.3f}' for v in vals) + f'|{ctc_torque:.3f}|')
lines += [
    '',
    '噪声升到 0.005 时 CTC 的误差增幅仍很小，但力矩 RMS 明显增大；精度优势必须与输入代价一起报告，不能只看 RMSE。',
    '',
    '## 论文正式接触实验平台（1.5× 校准夹爪）',
    '',
    '2026-10-31 决定：论文正式接触实验采用 1.5× 校准夹爪。只放大夹爪执行器增益/偏置，控制器参数、轨迹、时序、夹爪指令和评分标准不变。1× 原始 Menagerie 模型保留为敏感性/失败边界，不作为论文主表。',
    '',
    '正式批次：5 工况 × 3 方法 × 6 种子（seeds 31–36）= 90 回合。',
    '',
    '|工况|成功数/18|最终平面误差(mm，成功回合)|接触力峰值(N，成功回合)|搬运相对位移(mm，成功回合)|',
    '|---|---:|---:|---:|---:|',
]
for sc in ['nominal', 'heavy', 'slippery', 'pulse', 'low_friction']:
    rs = [r for r in grasp15 if r['scenario'] == sc]
    ok = [r for r in rs if r['success'] == 'True']
    vals = [np.mean([float(r[k]) * scale for r in ok]) if ok else float('nan') for k, scale in
            [('final_xy_error_m', 1000), ('peak_contact_force_N', 1), ('transport_relative_motion_m', 1000)]]
    lines.append(f'|{sc}|{len(ok)}/{len(rs)}|' + '|'.join('—' if np.isnan(v) else f'{v:.3f}' for v in vals) + '|')
lines += [
    '',
    '正式平台总体 72/90；名义、重物、0.35 摩擦、脉冲均 18/18，0.12 摩擦 0/18 作为失败边界。三种控制器成功数相同，说明差异来自夹持能力，不是控制器。',
    '',
    '### 按方法拆分（1.5× 正式平台）',
    '',
    '|工况|CTC|SMC|RBF-SMC|',
    '|---|---:|---:|---:|',
]
for sc in ['nominal', 'heavy', 'slippery', 'pulse', 'low_friction']:
    vals = []
    for method in METHODS:
        rs = [r for r in grasp15 if r['scenario'] == sc and r['method'] == method]
        vals.append(f"{sum(r['success'] == 'True' for r in rs)}/{len(rs)}")
    lines.append('|' + sc + '|' + '|'.join(vals) + '|')
lines += [
    '',
    '代表性视频：`results/grasp_transport_grip15/nominal_RBF_SMC_31.mp4`、`heavy_RBF_SMC_31.mp4`、`slippery_RBF_SMC_31.mp4`；完整对照见 `docs/夹爪力校准对照报告.md`。',
    '',
    '## 1× 原始模型敏感性（45 回合，不作为论文主表）',
    '',
    '此表为未校准的 1× Menagerie 夹爪，保留作为失败边界和敏感性分析。成功要求确实抬升、搬运阶段双侧接触占比超过 50%，并最终一秒满足目标区域、支撑高度、低速度和倾角阈值。',
    '',
    '|工况|方法|成功数/3|最终平面误差均值(mm，含失败)|接触力峰值均值(N)|搬运相对位移均值(mm)|',
    '|---|---|---:|---:|---:|---:|',
]
for sc in ['nominal', 'heavy', 'slippery', 'pulse', 'low_friction']:
    for method in METHODS:
        rs = [r for r in grasp if r['scenario'] == sc and r['method'] == method]
        count = sum(r['success'] == 'True' for r in rs)
        vals = [np.mean([float(r[k]) * s for r in rs]) for k, s in
                [('final_xy_error_m', 1000), ('peak_contact_force_N', 1), ('transport_relative_motion_m', 1000)]]
        lines.append(f'|{sc}|{method}|{count}/3|' + '|'.join(f'{v:.3f}' for v in vals) + '|')
lines += [
    '',
    '所有方法共享同一预定时序、夹爪指令和力矩上限；没有针对某一控制器改变抓取高度或摩擦。低摩擦工况同时设置物体和指尖摩擦，实际接触摩擦值已核验。',
    '',
    '扩展批次把每格样本从 3 个种子提高到 6 个（seeds 31–36，共 90 回合）：名义和脉冲各 18/18，重物、0.35 摩擦、0.12 摩擦各 0/18。4 s 慢抬升 90 回合的成功数与默认相同，只把下滑检测时刻推迟约 0.5 s；oracle 对中诊断同样没有救回重物/低摩擦（重物最大抬升从约 0.465 m 升到约 0.476 m 后仍下滑）。阶段检测器在默认、慢抬升和 oracle 对中批次均为 TP=54、FP=0、TN=36、FN=0；夹爪力 1.5 倍使重物和 0.35 摩擦从 0/18 变为 18/18，0.12 摩擦需约 4 倍力才 18/18。详见 `docs/抓取扩展与阶段检测.md`。',
    '',
    '## 1× 与 1.5× 直接对照',
    '',
    '1× 是原始 Menagerie 模型的敏感性边界，1.5× 是论文正式实验平台。两者均为 5 工况 × 3 方法 × 6 种子 = 90 回合；控制参数、轨迹和时序相同，仅夹爪执行器增益/偏置不同。',
    '',
    '|工况|1× 成功数|1.5× 成功数|1× 成功率|1.5× 成功率|',
    '|---|---:|---:|---:|---:|',
]
for sc in ['nominal', 'heavy', 'slippery', 'pulse', 'low_friction']:
    a = [r for r in grasp90 if r['scenario'] == sc]
    b = [r for r in grasp15 if r['scenario'] == sc]
    sa = sum(r['success'] == 'True' for r in a); sb = sum(r['success'] == 'True' for r in b)
    lines.append(f'|{sc}|{sa}/{len(a)}|{sb}/{len(b)}|{sa / len(a):.3f}|{sb / len(b):.3f}|')
lines += [
    '',
    '1× 总体 36/90，1.5× 总体 72/90；三种控制器在 1.5× 下成功数相同（重物和 0.35 摩擦各 18/18），说明改动解决的是夹持能力，不是某一控制器的优势。0.12 摩擦在 1.5× 下仍 0/18，保留为失败边界；4× 夹爪力才使其 18/18，仅作灵敏度。代表性视频为 `results/grasp_transport_grip15/nominal_RBF_SMC_31.mp4`、`heavy_RBF_SMC_31.mp4` 和 `slippery_RBF_SMC_31.mp4`，完整对照见 `docs/夹爪力校准对照报告.md`。',
    '',
    '## 抓取失败的独立夹爪容量试验',
    '',
    '`python/gripper_hold_test.py --sweep` 将手臂固定在抓取构型，方块先在桌面上闭合，再对物体施加逐渐增大的上拉力，直到方块相对手指滑动 5 mm。该试验排除手臂跟踪误差，只测夹持界面；方块在手指间居中放置，因此结果是理想化容量估计，不等于实际任务中的精确容量；实际任务还叠加手部跟踪偏差和物体初始偏移。下拉承载容量定义为 `上拉容量 - mg`。',
    '',
    '|摩擦系数|下拉容量均值(N)|范围(N)|0.1 kg 重量比|0.25 kg 重量比|',
    '|---|---:|---:|---:|---:|',
]
for fric, (avg, lo, hi) in cap_by_fric.items():
    lines.append(f'|{fric:.2f}|{avg:.3f}|[{lo:.3f}, {hi:.3f}]|{avg / weights[0.1]:.2f}|{avg / weights[0.25]:.2f}|')
lines += [
    '',
    '0.1 kg 与 0.25 kg 的重量分别为 {:.3f} N 和 {:.3f} N。重物在 μ=1 时容量/重量约 1.0–1.2，0.35 摩擦约 1.0–1.2，处于抬升加速度和接触不对称即可触发的临界区；0.12 摩擦仅约 0.5，必然滑脱。上拉法换算假设摩擦锥上下方向近似对称；实际接触几何可能使两者略有差异，因此容量用于比较数量级和失败域，不作为精确阈值。'.format(weights[0.1], weights[0.25]),
    '',
    '抓取阶段的关节误差范数（4.5–7 s）在各工况不超过约 3.6 mm，而重物和低摩擦工况在 5.0 s 已有双侧接触、随后物体相对手指持续下滑。因此不能把失败归因于手臂轨迹跟踪；优先改进夹爪行程/力、抓取几何和接触摩擦。',
    '',
    '## 验证与边界',
    '',
    '- 所有自由运动回合 `contact_steps=0`、`initial_contacts=0`；引擎警告计数为 0。',
    '- 45 个 1× 接触回合和 90 个 1.5× 回合均无引擎警告；实际指尖接触摩擦为 1.0 / 0.35 / 0.12，已由 `effective_finger_friction_min` 核验。',
    '- 控制器已统一到 `python/controllers.py`；`python/test_controller_regression.py` 检查共享实现与归档公式逐样本一致、自由/抓取回放与 CSV 一致、异常种子零接触。',
    '- 当前仍是预设时序、小位置偏差的方块任务；没有视觉、VLA、LIBERO 原生任务或真实硬件。',
    '- 连续时间条件性有界性推导不覆盖接触切换、冲击、饱和和离散权值投影；仿真成功不能替代证明。',
    '- 小样本 18/18 或 3/3 都不代表真实部署成功率；失败回合全部保留，没有为了结论删除或修改评分标准。',
    '',
    '## 尚未完成的工作',
    '',
    '- LIBERO/OpenVLA-OFT 环境、检查点、动作接口和在线闭环；本机未下载 VLA 权重、未运行推理、未租云服务器。',
    '- 完整阶段恢复闭环：检测器已验证，但“检测后重新对中/重新抓取并继续任务”尚未实现；慢抬升和 oracle 对中试验证明仅减速或仅对中无效。当前接触任务主结果采用 1.5× 校准夹爪，0.12 摩擦仍为失败边界。',
    '- 更多物体尺寸/姿态、更多种子与统计精度估计；当前每工况每方法 6 个种子，仍不足以支持精确成功率结论。',
    '- 最终论文的假设数值认证、离散/饱和条件讨论、参考文献核验与答辩材料。',
    '',
    '## 复现入口',
    '',
    '```bash',
    'cd project',
    '.venv/bin/python python/tune_controllers.py --out results/fair_tuning_v2',
    '.venv/bin/python python/test_observation_noise.py --out results/fair_tuning_v2',
    '.venv/bin/python python/grasp_transport.py --batch --video',
    '.venv/bin/python python/grasp_transport.py --batch --seeds 31,32,33,34,35,36 --out results/grasp_transport_grip15 --grip-scale 1.5 --video',
    '.venv/bin/python python/report_grip_comparison.py',
    '.venv/bin/python python/gripper_hold_test.py --sweep',
    '.venv/bin/python python/report_contact_stage.py',
    '.venv/bin/python python/test_controller_regression.py',
    '```',
    '',
]
(ROOT / 'docs/公平调参与接触实验结果.md').write_text('\n'.join(lines))

# Plots.
fig, ax = plt.subplots(figsize=(7, 4), layout='constrained')
labels = ['development', 'held-out', 'noise']
v1 = [sum(float(r['max_contacts']) > 0 for r in old_dev),
      sum(float(r['max_contacts']) > 0 for r in old_held),
      sum(float(r['max_contacts']) > 0 for r in old_noise)]
v2 = [sum(int(r['contact_steps']) > 0 for r in dev),
      sum(int(r['contact_steps']) > 0 for r in held),
      sum(int(r['contact_steps']) > 0 for r in noise)]
x = np.arange(3)
ax.bar(x - .2, v1, .4, label='v1 (before audit fix)', color='#c44')
ax.bar(x + .2, v2, .4, label='v2 (fixed setup)', color='#2a7')
ax.set_xticks(x, labels); ax.set_ylabel('Runs with any contact')
ax.set_title('Contact audit: before and after setup fix'); ax.grid(axis='y'); ax.legend()
fig.savefig(ROOT / 'results/analysis/contact_audit_v1_v2.png', dpi=160); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4), layout='constrained')
frics = list(cap_by_fric)
avg = [cap_by_fric[f][0] for f in frics]
err = np.array([[cap_by_fric[f][0] - cap_by_fric[f][1] for f in frics],
                [cap_by_fric[f][2] - cap_by_fric[f][0] for f in frics]])
ax.errorbar(np.arange(len(frics)), avg, yerr=err, marker='o', capsize=4, label='grip downward capacity')
ax.axhline(weights[0.1], ls='--', color='#666', label='0.1 kg weight')
ax.axhline(weights[0.25], ls=':', color='#222', label='0.25 kg weight')
ax.set_xticks(np.arange(len(frics)), [f'{f:.2f}' for f in frics])
ax.set_xlabel('finger/object friction coefficient'); ax.set_ylabel('Force (N)')
ax.set_title('Gripper pull-hold capacity vs friction'); ax.grid(); ax.legend(fontsize=8)
fig.savefig(GB / 'capacity_vs_friction.png', dpi=160); plt.close(fig)

fig, axs = plt.subplots(2, 2, figsize=(12, 7), layout='constrained')
for i, sc in enumerate(['nominal', 'low_friction']):
    for method in METHODS:
        a = np.load(CT / f'{sc}_{method}_31.npz')['data']
        axs[i, 0].plot(a[:, 0], a[:, 4], label=method)
        axs[i, 1].plot(a[:, 0], a[:, 8], label=method)
    axs[i, 0].set(title=sc + ' object height', xlabel='Time (s)', ylabel='Height (m)')
    axs[i, 1].set(title=sc + ' finger contacts', xlabel='Time (s)', ylabel='Contacting fingers')
    for ax in axs[i]:
        ax.grid(); ax.legend(fontsize=8)
fig.savefig(CT / 'contact_comparison.png', dpi=160); plt.close(fig)

# Checks and provenance.
for r in grasp + grasp15:
    expected = .35 if r['scenario'] == 'slippery' else (.12 if r['scenario'] == 'low_friction' else 1.)
    assert abs(float(r['effective_finger_friction_min']) - expected) < 1e-8
checks = {
    'fair_tuning_v2': {
        'development_runs': len(dev), 'heldout_runs': len(held),
        'noise_runs': len(noise),
        'selected_identical_to_v1': bool(selected_identical),
        'max_abs_ee_rmse_diff_m': float(max_ee_diff),
        'all_free_runs_zero_contacts': bool(all(int(r['contact_steps']) == 0 for r in dev + held + noise)),
        'all_free_runs_zero_initial_contacts': bool(all(int(r['initial_contacts']) == 0 for r in dev + held + noise)),
        'all_warning_counts_zero': bool(all(int(r['warning_count']) == 0 for r in dev + held + noise)),
    },
    'grasp_transport': {
        'runs': len(grasp),
        'successes': int(sum(r['success'] == 'True' for r in grasp)),
        'all_warning_counts_zero': bool(all(int(r['warning_count']) == 0 for r in grasp)),
        'effective_contact_friction_verified': True,
    },
    'grasp_transport_grip15': {
        'runs': len(grasp15), 'successes': int(sum(r['success'] == 'True' for r in grasp15)),
        'all_warning_counts_zero': bool(all(int(r['warning_count']) == 0 for r in grasp15)),
        'effective_contact_friction_verified': True,
    },
    'gripper_bench': {'pull_sweep_runs': len(cap), 'capacity_by_friction': cap_by_fric},
}
(FT / 'checks.json').write_text(json.dumps(checks, indent=2))
(CT / 'checks.json').write_text(json.dumps(checks['grasp_transport'], indent=2))
files = ['controllers.py', 'run_panda_local.py', 'grasp_transport.py',
         'tune_controllers.py', 'test_observation_noise.py',
         'test_controller_regression.py', 'gripper_hold_test.py',
         'diagnose_contacts.py', 'diagnose_grasp.py', 'analyze_extended_grasp.py', 'report_grip_comparison.py',
         'report_contact_stage.py']
prov = {
    'source_sha256': {name: hashlib.sha256((ROOT / 'python' / name).read_bytes()).hexdigest() for name in files},
    'selected_config': selected,
    'datasets': {
        'fair_tuning_v2': 'clean rerun after setup fix; 96 dev + 126 held-out + 27 noise',
        'fair_tuning_v1_original': 'pre-audit archive; contact rows present only in payload scenarios',
        'grasp_transport': '45 runs with shared controller, same physics as archived batch',
        'grasp_transport_v1_original': 'pre-refactor archive, bit-identical metrics',
        'grasp_transport_ext': 'six-seed default-lift batch with phase detector fields',
        'grasp_slow_lift_ext': 'six-seed 4 s slow-lift control',
        'grasp_oracle_aligned': 'six-seed oracle object-centering diagnostic (not deployable)',
        'grasp_grip15_ext': 'six-seed 1.5x gripper-force diagnostic',
        'grasp_transport_grip15': 'formal six-seed 1.5x gripper-force batch with video',
        'grasp_grip40_low': '4x gripper-force low-friction diagnostic',
        'gripper_bench': 'independent pull-hold capacity sweep',
    },
    'formal_contact_platform': {'path': 'results/grasp_transport_grip15', 'gripper_force_scale': 1.5,
                                'decision_date': '2026-10-31', 'original_model_sensitivity': 'results/grasp_transport_ext'},
    'regression_check': 'python/test_controller_regression.py',
}
(ROOT / 'results/analysis/local_completion_provenance.json').write_text(json.dumps(prov, indent=2))
print(json.dumps(checks, indent=2))
print('Report, plots, checks and provenance written')
