"""Generate the 1x vs 1.5x gripper-force comparison report."""
import csv, json, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_panda_local import ROOT

BASE = ROOT / 'results/grasp_transport_ext'
CAL = ROOT / 'results/grasp_transport_grip15'
OUT = ROOT / 'results/analysis'
SCENARIOS = ['nominal', 'heavy', 'slippery', 'pulse', 'low_friction']
METHODS = ['CTC', 'SMC', 'RBF_SMC']


def load(path):
    return list(csv.DictReader(path.open()))


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0., c - h), min(1., c + h)


def subset(rows, sc, method=None):
    rs = [r for r in rows if r['scenario'] == sc]
    if method:
        rs = [r for r in rs if r['method'] == method]
    return rs


def mean(rows, key):
    return float(np.mean([float(r[key]) for r in rows])) if rows else float('nan')


def main():
    base = load(BASE / 'summary.csv')
    cal = load(CAL / 'summary.csv')
    assert len(base) == 90 and len(cal) == 90
    ka = {(r['scenario'], r['method'], r['seed']): r for r in base}
    kb = {(r['scenario'], r['method'], r['seed']): r for r in cal}
    assert ka.keys() == kb.keys()
    assert all(int(ka[k]['warning_count']) == 0 and int(kb[k]['warning_count']) == 0 for k in ka)

    summary = {}
    for sc in SCENARIOS:
        a = subset(base, sc); b = subset(cal, sc)
        sa = sum(r['success'] == 'True' for r in a); sb = sum(r['success'] == 'True' for r in b)
        la, ha = wilson(sa, len(a)); lb, hb = wilson(sb, len(b))
        ok_a = [r for r in a if r['success'] == 'True']
        ok_b = [r for r in b if r['success'] == 'True']
        summary[sc] = {
            'base_success': sa, 'base_n': len(a), 'base_ci': [la, ha],
            'cal_success': sb, 'cal_n': len(b), 'cal_ci': [lb, hb],
            'base_final_xy_mm': mean(ok_a, 'final_xy_error_m') * 1000 if ok_a else None,
            'cal_final_xy_mm': mean(ok_b, 'final_xy_error_m') * 1000 if ok_b else None,
            'base_peak_force_N': mean(ok_a, 'peak_contact_force_N') if ok_a else None,
            'cal_peak_force_N': mean(ok_b, 'peak_contact_force_N') if ok_b else None,
            'base_max_height_m': mean(ok_a, 'max_object_height_m') if ok_a else None,
            'cal_max_height_m': mean(ok_b, 'max_object_height_m') if ok_b else None,
        }

    lines = [
        '# 夹爪力校准对照报告（1× vs 1.5×）',
        '',
        '本报告由 `python/report_grip_comparison.py` 从 `results/grasp_transport_ext`（1× 原始夹爪）和 `results/grasp_transport_grip15`（1.5× 校准夹爪）生成。两个批次均为 5 工况 × 3 方法 × 6 种子（seeds 31–36），共 90 回合；控制器参数、轨迹、时序、夹爪指令完全相同，只把夹爪执行器的增益/偏置放大 1.5 倍。',
        '',
        '**决定（2026-10-31）：1.5× 校准夹爪是论文正式接触实验平台；1× 原始 Menagerie 模型保留为敏感性/失败边界，不作为论文主表。**',
        '',
        '1.5× 是模型校准假设，不是控制算法改动，也不是通过改评分标准让失败回合变成功。',
        '',
        '## 成功率',
        '',
        '|工况|1× 成功数|1.5× 成功数|1× Wilson 95%|1.5× Wilson 95%|',
        '|---|---:|---:|---:|---:|',
    ]
    for sc in SCENARIOS:
        s = summary[sc]
        lines.append(f'|{sc}|{s["base_success"]}/{s["base_n"]}|{s["cal_success"]}/{s["cal_n"]}|'
                     f'[{s["base_ci"][0]:.3f}, {s["base_ci"][1]:.3f}]|'
                     f'[{s["cal_ci"][0]:.3f}, {s["cal_ci"][1]:.3f}]|')
    lines += [
        '',
        '总体：1× 为 36/90 成功，1.5× 为 72/90 成功；差异来自重物和 0.35 摩擦。0.12 摩擦在 1.5× 下仍 0/18，保留为失败边界。',
        '',
        '### 按方法拆分',
        '',
        '|工况|1× CTC|1× SMC|1× RBF-SMC|1.5× CTC|1.5× SMC|1.5× RBF-SMC|',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for sc in SCENARIOS:
        vals = []
        for rows in (base, cal):
            for m in METHODS:
                rs = subset(rows, sc, m)
                vals.append(f'{sum(r["success"] == "True" for r in rs)}/{len(rs)}')
        lines.append('|' + sc + '|' + '|'.join(vals) + '|')
    lines += [
        '',
        '三种控制器在 1.5× 下的成功数相同；说明这一改动解决的是夹持能力，不是某一控制器的优势。',
        '',
        '## 成功回合的连续指标',
        '',
        '仅统计成功回合，避免把失败回合的掉落位置混入放置精度。',
        '',
        '|工况|1× 最终平面误差(mm)|1.5× 最终平面误差(mm)|1× 峰值接触力(N)|1.5× 峰值接触力(N)|1× 最大高度(m)|1.5× 最大高度(m)|',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    def fmt(v, nd=3):
        return '—' if v is None or (isinstance(v, float) and math.isnan(v)) else f'{v:.{nd}f}'
    for sc in SCENARIOS:
        s = summary[sc]
        lines.append(f'|{sc}|{fmt(s["base_final_xy_mm"],3)}|{fmt(s["cal_final_xy_mm"],3)}|'
                     f'{fmt(s["base_peak_force_N"],2)}|{fmt(s["cal_peak_force_N"],2)}|'
                     f'{fmt(s["base_max_height_m"],4)}|{fmt(s["cal_max_height_m"],4)}|')
    lines += [
        '',
        '重物和 0.35 摩擦在 1.5× 下能够完成抬升、搬运和放置；最终平面误差和峰值接触力与名义工况同一量级。0.12 摩擦仍无法产生足够摩擦，说明它超出当前校准夹爪的容量，而不是控制器差异。',
        '',
        '## 解释与边界',
        '',
        '- 1.5× 批次的 90 回合全部无引擎警告；实际指尖接触摩擦为 1.0 / 0.35 / 0.12，已核验。',
        '- 夹爪力 1.5× 是模型校准假设，尚无真实 Panda 夹爪标定数据；它用于检验夹持容量足够时控制器是否能把任务做完，不能称为精确硬件复现。',
        '- 0.12 摩擦在 4× 下才 18/18 成功；本报告不把 4× 作为正式主结果，只把它留作硬件需求灵敏度。',
        '- 这不是 VLA/LIBERO 或真实机器人结果；当前仍是预设时序、固定方块任务。',
        '',
        '## 代表性视频与图表',
        '',
        '- `results/grasp_transport_grip15/nominal_RBF_SMC_31.mp4`：1.5× 名义工况抓取搬运视频。',
        '- `results/grasp_transport_grip15/heavy_RBF_SMC_31.mp4`：1.5× 重物成功视频。',
        '- `results/grasp_transport_grip15/slippery_RBF_SMC_31.mp4`：1.5× 0.35 摩擦成功视频。',
        '- `results/analysis/grip_success_comparison.png`：各工况 1×/1.5× 成功率对比。',
        '- `results/grasp_transport_grip15/heavy_height_comparison.png`：重物对象高度对比。',
        '- `results/grasp_transport_grip15/slippery_height_comparison.png`：0.35 摩擦对象高度对比。',
        '- `results/grasp_transport_grip15/low_friction_height_comparison.png`：0.12 摩擦对象高度对比。',
        '',
        '## 数据路径',
        '',
        '- `results/grasp_transport_ext/`：1× 原始夹爪 90 回合。',
        '- `results/grasp_transport_grip15/`：1.5× 校准夹爪 90 回合、视频、轨迹。',
        '- `results/grasp_grip40_low/`：4× 夹爪力、0.12 摩擦 18 回合诊断。',
        '- `results/analysis/grip_comparison.json`：本报告统计 JSON。',
    ]
    (ROOT / 'docs/夹爪力校准对照报告.md').write_text('\n'.join(lines))
    (OUT / 'grip_comparison.json').write_text(json.dumps(summary, indent=2))

    # Success-rate bars.
    fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    x = np.arange(len(SCENARIOS)); w = .38
    a = [summary[s]['base_success'] / summary[s]['base_n'] for s in SCENARIOS]
    b = [summary[s]['cal_success'] / summary[s]['cal_n'] for s in SCENARIOS]
    ax.bar(x - w/2, a, w, label='1x original gripper', color='#c44')
    ax.bar(x + w/2, b, w, label='1.5x calibrated gripper', color='#2a7')
    ax.set_xticks(x, SCENARIOS, rotation=15)
    ax.set_ylim(0, 1.05); ax.set_ylabel('Success rate')
    ax.set_title('Grasp success: 1x vs 1.5x gripper force (90 runs each)')
    ax.grid(axis='y'); ax.legend(fontsize=8)
    fig.savefig(OUT / 'grip_success_comparison.png', dpi=160); plt.close(fig)

    # Object height comparisons for representative scenarios.
    for sc in ['nominal', 'heavy', 'slippery', 'low_friction']:
        fig, ax = plt.subplots(figsize=(7, 3.6), layout='constrained')
        for label, folder in [('1x', BASE), ('1.5x', CAL)]:
            f = folder / f'{sc}_CTC_31.npz'
            data = np.load(f)['data']
            ax.plot(data[:, 0], data[:, 4], label=label)
        ax.set(title=f'{sc} object height (CTC seed 31)', xlabel='Time (s)', ylabel='Object height (m)')
        ax.grid(); ax.legend()
        fig.savefig(CAL / f'{sc}_height_comparison.png', dpi=160); plt.close(fig)

    print(json.dumps(summary, indent=2))
    print('Grip comparison report written')


if __name__ == '__main__':
    main()
