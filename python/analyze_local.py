from pathlib import Path
import csv,json,hashlib
import numpy as np
from scipy.stats import t as student_t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/analysis';OUT.mkdir(exist_ok=True)

def load(path):return list(csv.DictReader(path.open()))
def grouped(rows):
 groups={}
 for r in rows:groups.setdefault((r['scenario'],r['controller']),[]).append(float(r['ee_rmse_m'])*1000)
 return groups

def table(rows):
 lines=['|工况|方法|回合数|末端 RMSE 均值 ± 样本标准差 (mm)|均值的 95% t 区间 (mm)|','|---|---|---:|---:|---:|']
 for (sc,method),vals in grouped(rows).items():
  a=np.array(vals);sd=a.std(ddof=1);ci=student_t.ppf(.975,len(a)-1)*sd/np.sqrt(len(a))
  lines.append(f'|{sc}|{method}|{len(a)}|{a.mean():.3f} ± {sd:.3f}|[{a.mean()-ci:.3f}, {a.mean()+ci:.3f}]|')
 return '\n'.join(lines)

mat=load(ROOT/'results/local_suite/summary.csv');panda=load(ROOT/'results/panda_local/summary.csv')
assert len(mat)==240 and len(panda)==60
assert all(float(r['warning_count'])==0 and float(r['actuator_mapping_error'])<1e-8 for r in panda)
text=['# 本地阶段实验汇总','',
'本报告由 `python/analyze_local.py` 从原始 CSV 生成。MATLAB 为 8 工况 × 6 方法 × 5 初始状态种子，共 240 回合；另有 seed=1 的 48 个开发回合和 seed=11 的 48 个控制周期检查回合。Panda 关节参考为 4 工况 × 5 方法 × 3 种子，共 60 回合。所有数据均来自本机实际运行。',
'','## 结论范围','',
'样本只覆盖少量初始状态扰动和规定的有限参考，不代表任务成功率或跨任务泛化。t 区间描述这些随机初始状态下的均值不确定性，小样本正态近似未得到充分验证；它不是对所有物理工况的性能保证。',
'','## MATLAB 二连杆','',table(mat),'','## Panda 七关节参考','',table(panda)]
cartpath=ROOT/'results/panda_cartesian/summary.csv'
if cartpath.exists():
 cart=load(cartpath)
 if len(cart)==60:
  text+=['','## Panda 笛卡尔路径经逆运动学生成参考','',table(cart)]
  assert all(float(r['warning_count'])==0 for r in cart)

text+=['','## 配对差异与消融','']
for label,rows in [('MATLAB',mat),('Panda joint',panda)]:
 for sc in ['nominal','payload','combined']:
  selected={method:{int(r['seed']):float(r['ee_rmse_m'])*1000 for r in rows if r['scenario']==sc and r['controller']==method} for method in ['SMC','RBF_SMC']}
  diff=np.array([selected['SMC'][seed]-selected['RBF_SMC'][seed] for seed in selected['SMC']])
  text.append(f'- {label} / {sc}：SMC 减去 RBF-SMC 的配对 RMSE 差均值 {diff.mean():.3f} mm；正值表示本次 RBF-SMC 更低。')
text+=['','RBF_NO_ROBUST 在某些工况的末段误差可能更低；RBF_FROZEN 在前三秒与在线更新完全一致，之后停止更新。完整方法不保证每个指标都最好。CTC 和 PD 没有完成相同控制努力预算下的系统调参，不能把本轮名次推广成方法的一般优劣。',
'','## 数值和物理检查','']
a={(r['scenario'],r['controller']):r for r in mat if r['seed']=='11'}
b=load(ROOT/'results/local_suite_dt1ms/summary.csv')
diffs=[]
for r in b:
 if r['scenario']=='measurement_noise':continue
 old=float(a[(r['scenario'],r['controller'])]['ee_rmse_m']);new=float(r['ee_rmse_m'])
 diffs.append(abs(new-old)*1000)
text.append(f'- MATLAB 同 seed 控制周期由 2 ms 减为 1 ms，排除噪声工况后最大 RMSE 绝对差为 {max(diffs):.4f} mm。噪声样本序列随采样率变化，不能将其差异单独解释为积分误差。')
text.append('- Panda 全部关节参考回合均无 MuJoCo 警告，实际关节执行器力与输入力矩映射误差低于 1e-8；未产生接触。')
text.append('- 独立模型、Coriolis 极化公式和权值边界验证见 `results/local_checks.json`。')
text+=['','## 尚未完成的最终课题部分','',
'在线 VLA 推理、LIBERO 配套相机和动作接口、抓取接触、真实物体搬运和放置成功率、负载抓取切换、正式多任务统计仍未完成。本地 Menagerie Panda 不能被称作已复现 LIBERO。当前参考模块是离线小路径的六维逆运动学加样条验证，不是在线动作安全过滤器，未认证避障。',
'','## 本地运行环境','',
'Python 3.14 与 MuJoCo 3.17.0 位于项目独立虚拟环境；依赖见 requirements-local.lock.txt。该环境专供本地物理与控制实验，不与未来 OFT 的 Python 3.10 / robosuite 1.4.1 环境混用。']
(ROOT/'docs/本地阶段完成报告.md').write_text('\n'.join(text))
# Overview heatmap: absolute errors avoid hiding poor baselines in percent gains.
fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
for axis,rows,title in zip(axes,[mat,panda],['MATLAB two-link','MuJoCo Panda']):
 g=grouped(rows);scs=list(dict.fromkeys(r['scenario'] for r in rows));methods=list(dict.fromkeys(r['controller'] for r in rows))
 values=np.array([[np.mean(g[(sc,me)]) for me in methods] for sc in scs])
 im=axis.imshow(values,cmap='YlOrRd',aspect='auto');axis.set_xticks(range(len(methods)),methods,rotation=40,ha='right');axis.set_yticks(range(len(scs)),scs);axis.set_title(title+' / endpoint RMSE (mm)')
 for i in range(len(scs)):
  for j in range(len(methods)):axis.text(j,i,f'{values[i,j]:.1f}',ha='center',va='center',fontsize=8)
 fig.colorbar(im,ax=axis)
fig.savefig(OUT/'overview.png',dpi=160);plt.close(fig)
# Evidence manifest excludes environment and large third-party assets.
manifest={}
for folder in ['matlab','python','results']:
 for p in (ROOT/folder).rglob('*'):
  if p.is_file() and not p.name.startswith('._') and '__pycache__' not in str(p) and p.name!='manifest.json' and p.suffix in ['.py','.m','.csv','.json','.npz','.mat']:
   manifest[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
print('Wrote report, overview, manifest')
