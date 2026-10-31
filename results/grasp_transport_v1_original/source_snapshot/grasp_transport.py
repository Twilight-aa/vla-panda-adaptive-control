"""Contact-only scripted pick/place. No weld, teleport or object-state feedback."""
import argparse,csv,json,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import mujoco as mj
from scipy.spatial.transform import Rotation
from run_panda_local import ROOT,HOME,LIMIT,setup,bias

OUT=ROOT/'results/grasp_transport';OUT.mkdir(exist_ok=True)
PHASES=['approach','descend','close','lift','transport','lower','release','retreat','settle']
TIMES=np.array([0,2,4,5,7,10,12,13,15,17.])

def scene(mass,friction,offset):
 folder=ROOT/'vendor/mujoco_menagerie/franka_emika_panda'
 tree=ET.parse(folder/'panda.xml');r=tree.getroot()
 r.find('compiler').set('meshdir',str(folder/'assets'))
 r.remove(r.find('keyframe'))
 world=r.find('worldbody')
 ET.SubElement(world,'light',pos='0 0 2',dir='0 0 -1',diffuse='.8 .8 .8')
 ET.SubElement(world,'geom',name='floor',type='plane',size='2 2 .05',rgba='.2 .25 .3 1')
 ET.SubElement(world,'geom',name='table',type='box',pos='.5 0 .30',size='.28 .36 .05',rgba='.6 .5 .4 1')
 ET.SubElement(world,'geom',name='target',type='box',pos='.5 .18 .3505',size='.055 .055 .0005',rgba='.1 .8 .2 .6',contype='0',conaffinity='0')
 body=ET.SubElement(world,'body',name='object',pos=f'{.45+offset[0]} {offset[1]} .375')
 ET.SubElement(body,'freejoint',name='object_joint')
 ET.SubElement(body,'geom',name='object_geom',type='box',size='.025 .025 .025',mass=str(mass),rgba='.85 .15 .1 1',friction=f'{friction} .005 .0001',condim='4')
 m=mj.MjModel.from_xml_string(ET.tostring(r,encoding='unicode'));m.opt.timestep=.001
 m.actuator_gainprm[:7]=0;m.actuator_gainprm[:7,0]=1;m.actuator_biasprm[:7]=0
 m.actuator_ctrlrange[:7]=np.column_stack((-LIMIT,LIMIT))
 finger_bodies=[m.body('left_finger').id,m.body('right_finger').id]
 m.geom_friction[np.isin(m.geom_bodyid,finger_bodies),0]=friction
 d=mj.MjData(m);d.qpos[:7]=HOME;d.qpos[7:9]=.04;d.ctrl[7]=255;mj.mj_forward(m,d)
 return m,d

def make_plan():
 m,d=setup();d.qpos[:7]=HOME;mj.mj_forward(m,d);b=m.body('hand').id
 R=np.array([[0.,1,0],[1,0,0],[0,0,-1.]])
 home=d.xpos[b].copy()
 targets=[home,[.45,0,.60],[.45,0,.4734],[.45,0,.4734],[.45,0,.60],
          [.5,.18,.60],[.5,.18,.4734],[.5,.18,.4734],[.5,.18,.60],[.5,.18,.60]]
 q=HOME.copy();qs=[q.copy()]
 for pos in targets[1:]:
  for it in range(300):
   d.qpos[:7]=q;mj.mj_forward(m,d)
   e=np.r_[np.array(pos)-d.xpos[b],Rotation.from_matrix(R@d.xmat[b].reshape(3,3).T).as_rotvec()]
   if np.linalg.norm(e)<1e-5:break
   jp=np.zeros((3,m.nv));jr=jp.copy();mj.mj_jacBody(m,d,jp,jr,b);J=np.vstack([jp[:,:7],jr[:,:7]])
   inc=J.T@np.linalg.solve(J@J.T+1e-5*np.eye(6),e)
   q=np.clip(q+inc*min(1,.05/max(np.linalg.norm(inc),1e-12)),m.jnt_range[:7,0]+.01,m.jnt_range[:7,1]-.01)
  else:raise RuntimeError('IK failed')
  qs.append(q.copy())
 qs=np.array(qs)
 def ref(t):
  i=min(np.searchsorted(TIMES,t,side='right')-1,len(TIMES)-2);duration=TIMES[i+1]-TIMES[i]
  x=np.clip((t-TIMES[i])/duration,0,1);delta=qs[i+1]-qs[i]
  blend=10*x**3-15*x**4+6*x**5
  db=(30*x**2-60*x**3+30*x**4)/duration
  ddb=(60*x-180*x**2+120*x**3)/duration**2
  return qs[i]+delta*blend,delta*db,delta*ddb,i
 np.savez(OUT/'scripted_plan.npz',times=TIMES,q=qs,targets=targets)
 return ref

class Controller:
 def __init__(self,method,config):
  self.m,self.d=setup();self.method=method;self.c=config;self.W=np.zeros((7,33))
  self.centers=np.random.default_rng(20261031).uniform(-1,1,(33,21))
 def step(self,q,v,qd,dqd,ddqd):
  m,d=self.m,self.d;c=self.c;e=q[:7]-qd;de=v[:7]-dqd
  lam=c.get('lambda',8);s=de+lam*e;vr=np.zeros(9);ar=np.zeros(9);vr[:7]=dqd-lam*e;ar[:7]=ddqd-lam*de
  h=bias(m,d,q,v);passive=d.qfrc_passive.copy();M=np.zeros((9,9));mj.mj_fullM(m,d,M)
  if self.method=='CTC':
   a=np.zeros(9);a[:7]=ddqd-c.get('ctc_kd',16)*de-c.get('ctc_kp',64)*e
   tau=(M@a+h-passive)[:7]
  else:
   g=bias(m,d,q,np.zeros(9));cvr=.5*(bias(m,d,q,v+vr)-h-bias(m,d,q,vr)+g)
   Ks=np.array([20,20,15,15,2,2,1])*c.get('ks_scale',1)
   rho=np.array([3,3,2,2,.5,.5,.2])*c.get('rho_scale',1)
   tau=(M@ar+cvr+g-passive)[:7]-Ks*s-rho*np.clip(s/c.get('phi',.15),-1,1)
   if self.method=='RBF_SMC':
    z=np.r_[(q[:7]-HOME)/.4,v[:7],ar[:7]/5];phi=np.exp(-np.sum((self.centers-z)**2,axis=1)/32);phi/=max(np.linalg.norm(phi),1e-12)
    tau+=self.W@phi;self.W+=.001*(-c.get('gamma',8)*np.outer(s,phi)-.02*self.W)
    self.W*=min(1,50/max(np.linalg.norm(self.W),1e-12))
  return np.clip(tau,-LIMIT,LIMIT),np.any(np.abs(tau)>LIMIT)

def run(method='CTC',seed=1,scenario='nominal',video=False,config=None):
 rng=np.random.default_rng(seed);offset=np.zeros(2) if seed==1 else rng.uniform(-.004,.004,2)
 mass=.25 if scenario=='heavy' else .1;friction=.35 if scenario=='slippery' else (.12 if scenario=='low_friction' else 1.)
 m,d=scene(mass,friction,offset);ctrl=Controller(method,config or {});ref=make_plan()
 obj=m.body('object').id;hand=m.body('hand').id;objgeom=m.geom('object_geom').id
 fingerids={m.body('left_finger').id,m.body('right_finger').id}
 effective_friction=[]
 log=[];phase_max={p:0. for p in PHASES};writer=None;renderer=None
 if video:
  import imageio.v2 as imageio
  renderer=mj.Renderer(m,height=480,width=640);cam=mj.MjvCamera();cam.lookat[:]=[.45,.05,.4];cam.distance=1.65;cam.azimuth=135;cam.elevation=-25
  writer=imageio.get_writer(OUT/f'{scenario}_{method}_{seed}.mp4',fps=25)
 for k in range(17000):
  t=k*.001;qd,dqd,ddqd,phase=ref(t)
  u,sat=ctrl.step(d.qpos[:9].copy(),d.qvel[:9].copy(),qd,dqd,ddqd);d.ctrl[:7]=u
  if t<4:d.ctrl[7]=255
  elif t<5:d.ctrl[7]=255*(1-(t-4))
  elif t<12:d.ctrl[7]=0
  elif t<13:d.ctrl[7]=255*(t-12)
  else:d.ctrl[7]=255
  d.xfrc_applied[:]=0
  if scenario=='pulse' and 8<=t<8.15:d.xfrc_applied[obj,:3]=[1.5,0,0]
  mj.mj_step(m,d)
  maxforce=0.;touch=set()
  for ci in range(d.ncon):
   contact=d.contact[ci]
   if objgeom in [contact.geom1,contact.geom2]:
    force=np.zeros(6);mj.mj_contactForce(m,d,ci,force);maxforce=max(maxforce,float(np.linalg.norm(force[:3])))
    other=contact.geom2 if contact.geom1==objgeom else contact.geom1
    body=m.geom_bodyid[other]
    if body in fingerids:
     touch.add(body);effective_friction.append(float(contact.friction[0]))
  phase_max[PHASES[phase]]=max(phase_max[PHASES[phase]],maxforce)
  # Log object free-joint linear speed, pose, grip contacts, tracking and saturation.
  relative=d.xmat[hand].reshape(3,3).T@(d.xpos[obj]-d.xpos[hand])
  tilt=np.arccos(np.clip(d.xmat[obj].reshape(3,3)[2,2],-1,1))
  log.append(np.r_[t,phase,d.xpos[obj],d.qvel[9:12],len(touch),maxforce,np.linalg.norm(d.qpos[:7]-qd),sat,np.linalg.norm(ctrl.W),d.qpos[7:9],relative,tilt])
  if writer and k%40==0:
   renderer.update_scene(d,cam);frame=renderer.render();writer.append_data(frame)
   if k in [6000,9000,16000]:imageio.imwrite(OUT/f'{scenario}_{method}_{seed}_{k}.png',frame)
  if not np.isfinite(d.qpos).all():raise RuntimeError('Nonfinite state')
 if writer:writer.close();renderer.close()
 a=np.array(log);final=a[a[:,0]>=16];transport=a[(a[:,0]>=7)&(a[:,0]<10)]
 lifted=bool(np.max(a[:,4])>.425)
 placed=bool(np.all(np.linalg.norm(final[:,2:4]-[.5,.18],axis=1)<.04) and np.all(np.abs(final[:,4]-.375)<.01) and np.max(np.linalg.norm(final[:,5:8],axis=1))<.05 and np.max(final[:,18])<np.deg2rad(10))
 grasped=bool(lifted and np.mean(transport[:,8]==2)>.5)
 dropped=bool(lifted and np.any(transport[:,4]<.40))
 failure='none' if placed and grasped else ('grasp_or_lift' if not lifted else ('transport_drop' if dropped else 'placement'))
 result=dict(method=method,scenario=scenario,seed=seed,success=bool(placed and grasped),lifted=lifted,dropped=dropped,failure_phase=failure,
   final_xy_error_m=float(np.linalg.norm(final[-1,2:4]-[.5,.18])),max_object_height_m=float(a[:,4].max()),
   peak_contact_force_N=float(a[:,9].max()),saturation_fraction=float(a[:,11].mean()),warning_count=int(sum(w.number for w in d.warning)),
   transport_bilateral_contact_fraction=float(np.mean(transport[:,8]==2)),
   transport_relative_motion_m=float(np.max(np.linalg.norm(transport[:,15:18]-transport[0,15:18],axis=1))),
   final_tilt_deg=float(np.rad2deg(final[-1,18])),
   effective_finger_friction_min=float(min(effective_friction)) if effective_friction else -1.)
 stem=f'{scenario}_{method}_{seed}'
 np.savez_compressed(OUT/f'{stem}.npz',data=a,columns=['t','phase','object_x','object_y','object_z','vx','vy','vz','finger_contacts','max_object_contact_force','joint_error_norm','saturation','weight_norm','finger1','finger2','relative_x','relative_y','relative_z','tilt_rad'])
 (OUT/f'{stem}.json').write_text(json.dumps(dict(**result,phase_peak_contact_force=phase_max),indent=2));print(json.dumps(result),flush=True)
 return result

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--batch',action='store_true');ap.add_argument('--video',action='store_true');args=ap.parse_args()
 if args.batch:
  selected=json.loads((ROOT/'results/fair_tuning/selected.json').read_text());rows=[]
  for sc in ['nominal','heavy','slippery','pulse','low_friction']:
   for seed in [31,32,33]:
    for method in ['CTC','SMC','RBF_SMC']:
     rows.append(run(method,seed,sc,args.video and seed==31 and sc=='nominal' and method=='RBF_SMC',selected[method]))
     with (OUT/'summary.csv').open('w') as f:
      w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 else:run(video=args.video)
if __name__=='__main__':main()
