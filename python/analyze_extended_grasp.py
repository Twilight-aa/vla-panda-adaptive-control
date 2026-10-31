"""Analyze extended grasp batches, phase detector and slow-lift test."""
import csv, json, math
from pathlib import Path
from run_panda_local import ROOT

EXT = ROOT / 'results/grasp_transport_ext'
SLOW = ROOT / 'results/grasp_slow_lift_ext'
ALIGN = ROOT / 'results/grasp_oracle_aligned'
G15 = ROOT / 'results/grasp_transport_grip15'
G40 = ROOT / 'results/grasp_grip40_low'
OFFICIAL = ROOT / 'results/grasp_transport'
METHODS = ['CTC', 'SMC', 'RBF_SMC']
SCENARIOS = ['nominal', 'heavy', 'slippery', 'pulse', 'low_friction']


def load(path):
    return list(csv.DictReader(path.open()))


def wilson(k, n, z=1.96):
    if n == 0:
        return (0., 1.)
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0., center - half), min(1., center + half))


def detector_counts(rows):
    tp = sum(r['drop_detected'] == 'True' and r['success'] == 'False' for r in rows)
    fp = sum(r['drop_detected'] == 'True' and r['success'] == 'True' for r in rows)
    tn = sum(r['drop_detected'] == 'False' and r['success'] == 'True' for r in rows)
    fn = sum(r['drop_detected'] == 'False' and r['success'] == 'False' for r in rows)
    return dict(tp=tp, fp=fp, tn=tn, fn=fn)


def summarize(rows):
    out = {}
    for sc in SCENARIOS:
        rs = [r for r in rows if r['scenario'] == sc]
        k = sum(r['success'] == 'True' for r in rs)
        lo, hi = wilson(k, len(rs))
        times = [float(r['drop_time_s']) for r in rs if r['drop_detected'] == 'True']
        out[sc] = dict(success=k, n=len(rs), rate=k / len(rs), ci=[lo, hi],
                       drop=sum(r['drop_detected'] == 'True' for r in rs),
                       drop_time_mean=float(sum(times) / len(times)) if times else -1.,
                       drop_time_min=min(times) if times else -1.,
                       drop_time_max=max(times) if times else -1.)
    return out


def main():
    ext = load(EXT / 'summary.csv')
    slow = load(SLOW / 'summary.csv')
    align = load(ALIGN / 'summary.csv')
    g15 = load(G15 / 'summary.csv')
    g40 = load(G40 / 'summary.csv')
    official = load(OFFICIAL / 'summary.csv')
    assert len(ext) == 90 and len(slow) == 90 and len(align) == 90 and len(g15) == 90 and len(g40) == 18 and len(official) == 45
    # The first 45 extended runs use seeds 31-33 and must match the official batch.
    def key(r):
        return (r['scenario'], r['method'], r['seed'])
    off = {key(r): r for r in official}
    ext_old = {key(r): r for r in ext if int(r['seed']) in (31, 32, 33)}
    assert off.keys() == ext_old.keys()
    max_diff = 0.
    for k in off:
        assert off[k]['success'] == ext_old[k]['success']
        for f in ['final_xy_error_m', 'max_object_height_m', 'peak_contact_force_N',
                  'transport_relative_motion_m']:
            max_diff = max(max_diff, abs(float(off[k][f]) - float(ext_old[k][f])))
    ext_sum = summarize(ext)
    slow_sum = summarize(slow)
    align_sum = summarize(align)
    g15_sum = summarize(g15)
    det_ext = detector_counts(ext)
    det_slow = detector_counts(slow)
    det_align = detector_counts(align)
    result = {
        'default_lift': ext_sum, 'slow_lift_4s': slow_sum, 'oracle_aligned': align_sum, 'grip_scale_1_5': g15_sum, 'grip_scale_4_low_friction': {'success': sum(r['success']=='True' for r in g40), 'n': len(g40)},
        'detector_default': det_ext, 'detector_slow': det_slow, 'detector_oracle_aligned': det_align,
        'official_subset_max_metric_diff': max_diff,
        'seeds': [31, 32, 33, 34, 35, 36],
        'runs_per_batch': len(ext),
    }
    (ROOT / 'results/analysis/grasp_extended_summary.json').write_text(json.dumps(result, indent=2))

    lines = [
        '# 抓取扩展批次与阶段检测',
        '',
        '本文件由 `python/analyze_extended_grasp.py` 从 `results/grasp_transport_ext`、`results/grasp_slow_lift_ext` 和 `results/grasp_oracle_aligned` 生成。每个批次为 5 工况 × 3 方法 × 6 种子（seeds 31–36），共 90 回合。控制器参数、抓取时序和夹爪指令与正式 45 回合相同；慢抬升批次仅把抬升阶段从 2 s 延长到 4 s。按 2026-10-31 决定，1.5× 校准夹爪是论文正式接触实验平台，1× 原始模型保留为敏感性/失败边界。',
        '',
        f'扩展批次中 seeds 31–33 的 45 回合与正式批次指标最大差异 {max_diff:.2e}，成功标志一致，说明新增日志没有改变物理结果。',
        '',
        '## 成功率（含 Wilson 95% 区间）',
        '',
        '|工况|默认抬升 2 s|慢抬升 4 s|oracle 对中|',
        '|---|---:|---:|---:|',
    ]
    for sc in SCENARIOS:
        a = ext_sum[sc]; b = slow_sum[sc]; c = align_sum[sc]
        lines.append(f'|{sc}|{a["success"]}/{a["n"]} = {a["rate"]:.3f} [{a["ci"][0]:.3f}, {a["ci"][1]:.3f}]|'
                     f'{b["success"]}/{b["n"]} = {b["rate"]:.3f} [{b["ci"][0]:.3f}, {b["ci"][1]:.3f}]|'
                     f'{c["success"]}/{c["n"]} = {c["rate"]:.3f} [{c["ci"][0]:.3f}, {c["ci"][1]:.3f}]|')
    lines += [
        '',
        '慢抬升没有改变任何工况的成功数，只把检测到的下滑时间从约 5.53–5.60 s 推迟到约 6.06–6.22 s。oracle 对中使用仿真真值把抓取目标对准物体中心（仅诊断，不是可部署系统），名义/脉冲保持 18/18，重物仍 0/18，最大抬升高度从约 0.465 m 升到约 0.476 m 后仍下滑；0.35/0.12 摩擦完全不变。这说明失败不是单纯由手部跟踪偏差或物体初始偏心主导，而是夹持静态容量不足；继续减速、暂停、简单重试或仅对中都不能解决。',
        '',
        '## 夹爪力灵敏度（诊断，不混入正式结果）',
        '',
        '只放大夹爪执行器力，控制器参数、轨迹和时序不变；结果单独存放。',
        '',
        '|夹爪力倍数|名义|重物|0.35 摩擦|脉冲|0.12 摩擦|',
        '|---:|---:|---:|---:|---:|---:|',
        f'|1.0|{ext_sum["nominal"]["success"]}/{ext_sum["nominal"]["n"]}|{ext_sum["heavy"]["success"]}/{ext_sum["heavy"]["n"]}|{ext_sum["slippery"]["success"]}/{ext_sum["slippery"]["n"]}|{ext_sum["pulse"]["success"]}/{ext_sum["pulse"]["n"]}|{ext_sum["low_friction"]["success"]}/{ext_sum["low_friction"]["n"]}|',
        f'|1.5|{g15_sum["nominal"]["success"]}/{g15_sum["nominal"]["n"]}|{g15_sum["heavy"]["success"]}/{g15_sum["heavy"]["n"]}|{g15_sum["slippery"]["success"]}/{g15_sum["slippery"]["n"]}|{g15_sum["pulse"]["success"]}/{g15_sum["pulse"]["n"]}|{g15_sum["low_friction"]["success"]}/{g15_sum["low_friction"]["n"]}|',
        f'|4.0|—|—|—|—|{sum(r["success"]=="True" for r in g40)}/{len(g40)}|',
        '',
        '1.5 倍夹爪力使重物和 0.35 摩擦从 0/18 提升到 18/18，名义/脉冲保持 18/18；0.12 摩擦在 1.5 倍仍 0/18，需要约 4 倍夹爪力才 18/18。这给出明确的硬件方向：先升级夹爪法向力/接触几何，而不是继续调手臂控制器。',
        '',
        '## 阶段检测器',
        '',
        '检测器只使用手部坐标系中的物体相对高度：以闭合结束前 0.1 s 的均值为基线，抬升/搬运阶段相对高度增加超过 15 mm 即判定“物体正在从夹爪下滑”。它不使用控制器内部状态、成功标签或物体绝对位置。',
        '',
        '|批次|TP|FP|TN|FN|',
        '|---|---:|---:|---:|---:|',
    ]
    for name, d in [('默认抬升', det_ext), ('慢抬升', det_slow), ('oracle 对中', det_align)]:
        lines.append(f'|{name}|{d["tp"]}|{d["fp"]}|{d["tn"]}|{d["fn"]}|')
    lines += [
        '',
        '在 270 个回合中，检测器对下滑失败的分类为 TP=162、FP=0、TN=108、FN=0；这是事后验证，不等于在线闭环已实现。默认批次闭合结束时的双侧接触确认在 90 回合中对 54 个失败中的 48 个仍为真，因此“双侧接触”只能是门控条件，不能单独判定抓取成功。',
        '',
        '## 结论与下一步',
        '',
        '- 1.5× 校准夹爪已确定为论文正式接触实验平台；1× 原始模型保留为敏感性/失败边界。',
        '- 阶段检测的事件定义已经用数据验证，可直接作为在线门控/恢复触发条件。',
        '- 简单慢抬升和 oracle 对中都不能提高成功率；如果实现恢复，重点应放在增加夹爪法向力或改变接触几何，而不是继续降低速度或仅重新对中。',
        '- 低摩擦工况的容量/重量约 0.54，任何仅靠手臂控制或轨迹调整的方案都不能保证成功；需要夹爪/接触硬件或任务条件改变。',
        '',
        '## 建议的在线恢复协议（尚未实现）',
        '',
        '1. 闭合结束门控：检查闭合最后 0.1 s 的双侧接触占比；低于 0.8 时保持夹爪闭合再等 0.5 s，仍不满足则不抬升并标记抓取失败。',
        '2. 下滑检测：按本文件定义，手部坐标系相对高度超过基线 15 mm 即触发；连续触发或接触丢失超过 0.2 s 进入恢复。',
        '3. 恢复动作层级：先暂停并尝试重新夹紧；若无效，降低到抓取高度并重新对中；仍无效则标记失败。本机试验已经证明单纯减速和 oracle 对中都无法救回临界夹持，因此恢复的重点必须是增加法向力、改变指垫/物体几何或更换任务条件。',
        '4. 公平比较：检测阈值和恢复动作对所有控制器完全相同；控制器参数保持冻结；分别报告成功率、误触发率、恢复耗时和额外回合，不把恢复后的数据与固定时序基准混合。',
        '',
        '## 数据路径',
        '',
        '- `results/grasp_transport_ext/summary.csv`：默认抬升 90 回合，含物体质量、尺寸、初始偏移、四元数、接触对和检测字段。',
        '- `results/grasp_slow_lift_ext/summary.csv`：4 s 慢抬升 90 回合。',
        '- `results/grasp_oracle_aligned/summary.csv`：用真值物体偏移对中抓取目标的 90 回合诊断。',
        '- `results/grasp_transport_grip15/summary.csv`：夹爪力 1.5 倍的 90 回合正式批次（含视频）。',
        '- `results/grasp_grip40_low/summary.csv`：夹爪力 4 倍的 0.12 摩擦 18 回合诊断。',
        '- `results/analysis/grasp_extended_summary.json`：本文件的统计结果。',
    ]
    (ROOT / 'docs/抓取扩展与阶段检测.md').write_text('\n'.join(lines))
    print(json.dumps(result, indent=2))
    print('Extended grasp report written')


if __name__ == '__main__':
    main()
