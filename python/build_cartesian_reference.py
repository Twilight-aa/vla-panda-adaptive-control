"""Offline six-dimensional IK + smooth joint reference; synthetic inputs, no VLA."""
import json
import numpy as np
import mujoco as mj
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation
from run_panda_local import setup,HOME,ROOT

m,d=setup(); hand=m.body('hand').id;d.qpos[:7]=HOME;mj.mj_forward(m,d)
p0=d.xpos[hand].copy(); R0=d.xmat[hand].reshape(3,3).copy()
q=HOME.copy(); ts=np.linspace(0,6,61); qs=[]; residuals=[]
for t in ts:
 w=2*np.pi/6
 target=p0+np.array([.02*np.sin(w*t),.015*(1-np.cos(w*t)),.01*np.sin(2*w*t)])
 for iteration in range(200):
  d.qpos[:7]=q;mj.mj_forward(m,d)
  pos=target-d.xpos[hand]
  rot=Rotation.from_matrix(R0@d.xmat[hand].reshape(3,3).T).as_rotvec()
  err=np.r_[pos,rot]
  if np.linalg.norm(pos)<1e-6 and np.linalg.norm(rot)<1e-5:break
  jp=np.zeros((3,m.nv));jr=jp.copy();mj.mj_jacBody(m,d,jp,jr,hand)
  J=np.vstack([jp[:,:7],jr[:,:7]])
  delta=J.T@np.linalg.solve(J@J.T+1e-5*np.eye(6),err)
  q=np.clip(q+delta*min(1,.04/max(np.linalg.norm(delta),1e-12)),m.jnt_range[:7,0]+.01,m.jnt_range[:7,1]-.01)
 else: raise RuntimeError(f'IK did not converge at {t}')
 qs.append(q.copy());residuals.append([np.linalg.norm(pos),np.linalg.norm(rot)])
qs=np.array(qs); spline=CubicSpline(ts,qs,bc_type='natural');dense=np.linspace(0,6,6001)
vmax=np.max(np.abs(spline(dense,1)));amax=np.max(np.abs(spline(dense,2)))
scale=max(1,vmax/.5,np.sqrt(amax/2))
ts*=scale;spline=CubicSpline(ts,qs,bc_type='natural');dense=np.linspace(0,ts[-1],6001)
Q=spline(dense);V=spline(dense,1);A=spline(dense,2)
assert np.max(np.abs(V))<=.50001 and np.max(np.abs(A))<=2.00001
assert np.all(Q>=m.jnt_range[:7,0]) and np.all(Q<=m.jnt_range[:7,1])
# Verify interpolated path between knots, not only IK targets.
poserrs=[];roterrs=[]
for j in range(0,len(dense),10):
 t=dense[j]/scale;d.qpos[:7]=Q[j];mj.mj_forward(m,d)
 target=p0+np.array([.02*np.sin(w*t),.015*(1-np.cos(w*t)),.01*np.sin(2*w*t)])
 poserrs.append(np.linalg.norm(target-d.xpos[hand]))
 roterrs.append(np.linalg.norm(Rotation.from_matrix(R0@d.xmat[hand].reshape(3,3).T).as_rotvec()))
out=ROOT/'results/cartesian_reference';out.mkdir(exist_ok=True)
np.savez_compressed(out/'reference.npz',t=dense,q=Q,dq=V,ddq=A)
report={'duration_s':float(ts[-1]),'time_scale':float(scale),'max_velocity_rad_s':float(np.max(np.abs(V))),
 'max_acceleration_rad_s2':float(np.max(np.abs(A))),
 'max_interpolated_position_error_m':float(max(poserrs)),
 'max_interpolated_orientation_error_rad':float(max(roterrs)),
 'joint_limits_ok':True,'collision_avoidance_certified':False,
 'note':'Synthetic offline Cartesian reference; natural spline starts with nonzero velocity. Match initial velocity or add a startup blend before hardware use.'}
(out/'checks.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
