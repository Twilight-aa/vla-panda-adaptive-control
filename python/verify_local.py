"""Physical and experiment integrity checks; writes machine-readable evidence."""
from pathlib import Path
import csv,json
import numpy as np
import mujoco as mj
from run_panda_local import setup,bias,HOME,ROOT

m,d=setup(); q=d.qpos.copy(); q[:7]=HOME
v=np.array([.1,-.2,.15,.1,-.1,.2,.1,0,0]); vr=.7*v[::-1]
def matrix(q):
 d.qpos[:]=q;d.qvel[:]=0;mj.mj_forward(m,d)
 M=np.zeros((m.nv,m.nv));mj.mj_fullM(m,d,M);return M
M=matrix(q); derivatives=[]
for k in range(m.nv):
 dq=np.zeros(m.nq);dq[k]=1e-6
 derivatives.append((matrix(q+dq)-matrix(q-dq))/(2e-6))
D=np.array(derivatives)
C=np.zeros_like(M)
for i in range(m.nv):
 for j in range(m.nv):
  for k in range(m.nv):
   C[i,j]+=.5*(D[k,i,j]+D[j,i,k]-D[i,j,k])*v[k]
h=bias(m,d,q,v);g=bias(m,d,q,np.zeros(m.nv))
polar=.5*(bias(m,d,q,v+vr)-h-bias(m,d,q,vr)+g)
error=float(np.max(np.abs(polar-C@vr)))
assert error<1e-6, error
actual,ad=setup(.5,3)
assert actual.body('hand').mass[0]>m.body('hand').mass[0]
assert np.allclose(actual.dof_damping[:7],3*m.dof_damping[:7])
assert np.all(m.actuator_gainprm[:7,0]==1) and np.all(m.actuator_biasprm[:7]==0)
summary=list(csv.DictReader((ROOT/'results/local_suite/summary.csv').open()))
assert len(summary)==240
assert len({(r['seed'],r['scenario'],r['controller']) for r in summary})==240
assert all(np.isfinite(float(r['ee_rmse_m'])) for r in summary)
assert all(float(r['max_weight_norm'])<=50+1e-9 for r in summary)
# Early warmup of frozen ablation must be identical to online controller.
for scenario in ['nominal','payload','combined']:
 a=np.loadtxt(ROOT/f'results/local_suite/{scenario}_RBF_SMC.csv',delimiter=',',skiprows=1)
 b=np.loadtxt(ROOT/f'results/local_suite/{scenario}_RBF_FROZEN.csv',delimiter=',',skiprows=1)
 assert np.allclose(a[a[:,0]<3],b[b[:,0]<3])
panda_count=0
for folder in ['panda_local','panda_cartesian']:
 rows=list(csv.DictReader((ROOT/f'results/{folder}/summary.csv').open()))
 assert len(rows)==60
 assert len({(r['seed'],r['scenario'],r['controller']) for r in rows})==60
 assert all(float(r['warning_count'])==0 for r in rows)
 assert all(float(r['actuator_mapping_error'])<1e-8 for r in rows)
 assert all(float(r['max_weight_norm'])<=50+1e-9 for r in rows)
 panda_count+=len(rows)
checks={'panda_completed_unique_runs_across_two_references':panda_count,'panda_min_inertia_eigenvalue':float(np.linalg.eigvalsh(M).min()),
 'coriolis_polarization_error':error,'independent_nominal_model':True,
 'torque_actuator_configuration':True,'matlab_completed_unique_runs':len(summary),
 'matlab_weights_within_cap':True,'frozen_warmup_matches_online':True}
(ROOT/'results/local_checks.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
