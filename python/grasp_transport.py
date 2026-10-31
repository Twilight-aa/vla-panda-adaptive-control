"""Contact-only scripted pick/place. No weld, teleport or object-state feedback."""
import argparse,csv,json,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import mujoco as mj
from scipy.spatial.transform import Rotation
from run_panda_local import ROOT,HOME,LIMIT,setup,bias
from controllers import TorqueController as Controller

OUT=ROOT/'results/grasp_transport';OUT.mkdir(exist_ok=True)
PHASES=['approach','descend','close','lift','transport','lower','release','retreat','settle']
TIMES=np.array([0,2,4,5,7,10,12,13,15,17.])

def schedule(lift_duration=2.):
    lift_end=5.+lift_duration
    return np.array([0.,2.,4.,5.,lift_end,lift_end+3,lift_end+5,lift_end+6,lift_end+8,lift_end+10])

def scene(mass,friction,offset,grip_scale=1.):
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
 if grip_scale!=1.:
  m.actuator_gainprm[7,0]*=grip_scale;m.actuator_biasprm[7,1]*=grip_scale;m.actuator_biasprm[7,2]*=grip_scale
 m.actuator_gainprm[:7]=0;m.actuator_gainprm[:7,0]=1;m.actuator_biasprm[:7]=0
 m.actuator_ctrlrange[:7]=np.column_stack((-LIMIT,LIMIT))
 finger_bodies=[m.body('left_finger').id,m.body('right_finger').id]
 m.geom_friction[np.isin(m.geom_bodyid,finger_bodies),0]=friction
 d=mj.MjData(m);d.qpos[:7]=HOME;d.qpos[7:9]=.04;d.ctrl[7]=255;mj.mj_forward(m,d)
 return m,d

def make_plan(z_offset=0.,outdir=None,lift_duration=2.,x_offset=0.,y_offset=0.):
 out=OUT if outdir is None else Path(outdir);out.mkdir(parents=True,exist_ok=True)
 times=schedule(lift_duration)
 m,d=setup();d.qpos[:7]=HOME;mj.mj_forward(m,d);b=m.body('hand').id
 R=np.array([[0.,1,0],[1,0,0],[0,0,-1.]])
 home=d.xpos[b].copy()
 grasp_z=.4734+z_offset
 targets=[home,[.45+x_offset,y_offset,.60],[.45+x_offset,y_offset,grasp_z],[.45+x_offset,y_offset,grasp_z],[.45+x_offset,y_offset,.60],
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
  i=min(np.searchsorted(times,t,side='right')-1,len(times)-2);duration=times[i+1]-times[i]
  x=np.clip((t-times[i])/duration,0,1);delta=qs[i+1]-qs[i]
  blend=10*x**3-15*x**4+6*x**5
  db=(30*x**2-60*x**3+30*x**4)/duration
  ddb=(60*x-180*x**2+120*x**3)/duration**2
  return qs[i]+delta*blend,delta*db,delta*ddb,i
 suffix='' if lift_duration==2. else f'_L{lift_duration:g}'
 if abs(x_offset)>1e-12 or abs(y_offset)>1e-12:suffix+=f'_aligned_{x_offset:+.4f}_{y_offset:+.4f}'
 name='scripted_plan.npz' if abs(z_offset)<1e-12 and not suffix else f'scripted_plan_z{z_offset:+.4f}{suffix}.npz'
 np.savez(out/name,times=times,q=qs,targets=targets)
 return ref

def run(method='CTC',seed=1,scenario='nominal',video=False,config=None,
        mass=None,friction=None,z_offset=0.,outdir=None,lift_duration=2.,align_object=False,grip_scale=1.):
 out=OUT if outdir is None else Path(outdir);out.mkdir(parents=True,exist_ok=True)
 times=schedule(lift_duration)
 rng=np.random.default_rng(seed);offset=np.zeros(2) if seed==1 else rng.uniform(-.004,.004,2)
 mass=(.25 if scenario=='heavy' else .1) if mass is None else mass
 friction=(.35 if scenario=='slippery' else (.12 if scenario=='low_friction' else 1.)) if friction is None else friction
 px,py=(offset if align_object else (0.,0.))
 m,d=scene(mass,friction,offset,grip_scale);ctrl=Controller(method,config or {});ref=make_plan(z_offset,out,lift_duration,px,py)
 obj=m.body('object').id;hand=m.body('hand').id;objgeom=m.geom('object_geom').id
 fingerids={m.body('left_finger').id,m.body('right_finger').id}
 effective_friction=[]
 log=[];phase_max={p:0. for p in PHASES};writer=None;renderer=None;contact_pairs=set()
 if video:
  import imageio.v2 as imageio
  renderer=mj.Renderer(m,height=480,width=640);cam=mj.MjvCamera();cam.lookat[:]=[.45,.05,.4];cam.distance=1.65;cam.azimuth=135;cam.elevation=-25
  writer=imageio.get_writer(out/f'{scenario}_{method}_{seed}.mp4',fps=25)
 for k in range(int(round(times[-1]/.001))):
  t=k*.001;qd,dqd,ddqd,phase=ref(t)
  u,sat,_=ctrl.step(d.qpos[:9].copy(),d.qvel[:9].copy(),qd,dqd,ddqd);d.ctrl[:7]=u
  if t<times[2]:d.ctrl[7]=255
  elif t<times[3]:d.ctrl[7]=255*(1-(t-times[2])/(times[3]-times[2]))
  elif t<times[6]:d.ctrl[7]=0
  elif t<times[7]:d.ctrl[7]=255*(t-times[6])/(times[7]-times[6])
  else:d.ctrl[7]=255
  d.xfrc_applied[:]=0
  if scenario=='pulse' and times[4]+1<=t<times[4]+1.15:d.xfrc_applied[obj,:3]=[1.5,0,0]
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
     contact_pairs.add(tuple(sorted((m.body(m.geom_bodyid[objgeom]).name,m.body(body).name))))
  phase_max[PHASES[phase]]=max(phase_max[PHASES[phase]],maxforce)
  # Log object free-joint linear speed, pose, grip contacts, tracking and saturation.
  relative=d.xmat[hand].reshape(3,3).T@(d.xpos[obj]-d.xpos[hand])
  tilt=np.arccos(np.clip(d.xmat[obj].reshape(3,3)[2,2],-1,1))
  log.append(np.r_[t,phase,d.xpos[obj],d.qvel[9:12],len(touch),maxforce,np.linalg.norm(d.qpos[:7]-qd),sat,np.linalg.norm(ctrl.W),d.qpos[7:9],relative,tilt,d.xquat[obj]])
  if writer and k%40==0:
   renderer.update_scene(d,cam);frame=renderer.render();writer.append_data(frame)
   if k in [6000,9000,16000]:imageio.imwrite(out/f'{scenario}_{method}_{seed}_{k}.png',frame)
  if not np.isfinite(d.qpos).all():raise RuntimeError('Nonfinite state')
 if writer:writer.close();renderer.close()
 a=np.array(log);final=a[a[:,0]>=times[-1]-1];transport=a[(a[:,0]>=times[4])&(a[:,0]<times[5])]
 lifted=bool(np.max(a[:,4])>.425)
 placed=bool(np.all(np.linalg.norm(final[:,2:4]-[.5,.18],axis=1)<.04) and np.all(np.abs(final[:,4]-.375)<.01) and np.max(np.linalg.norm(final[:,5:8],axis=1))<.05 and np.max(final[:,18])<np.deg2rad(10))
 grasped=bool(lifted and np.mean(transport[:,8]==2)>.5)
 dropped=bool(lifted and np.any(transport[:,4]<.40))
 failure='none' if placed and grasped else ('grasp_or_lift' if not lifted else ('transport_drop' if dropped else 'placement'))
 # Post-hoc phase detector: both fingers on the object at the end of close,
 # and object slipping away from the hand during lift/transport.
 close=a[(a[:,0]>=4.9)&(a[:,0]<=5.0)]
 rel_grasp=float(np.mean(close[:,17]))
 bilateral_close=float(np.mean(close[:,8]==2))
 grasp_confirmed=bool(bilateral_close>=.8)
 post=a[(a[:,0]>=times[3]+.5)&(a[:,0]<=times[6])]
 excess=post[:,17]-rel_grasp
 drop_idx=np.where(excess>.015)[0]
 drop_detected=bool(drop_idx.size>0)
 drop_time=float(post[drop_idx[0],0]) if drop_detected else -1.
 result=dict(method=method,scenario=scenario,seed=seed,success=bool(placed and grasped),lifted=lifted,dropped=dropped,failure_phase=failure,
   final_xy_error_m=float(np.linalg.norm(final[-1,2:4]-[.5,.18])),max_object_height_m=float(a[:,4].max()),
   peak_contact_force_N=float(a[:,9].max()),saturation_fraction=float(a[:,11].mean()),warning_count=int(sum(w.number for w in d.warning)),
   transport_bilateral_contact_fraction=float(np.mean(transport[:,8]==2)),
   transport_relative_motion_m=float(np.max(np.linalg.norm(transport[:,15:18]-transport[0,15:18],axis=1))),
   final_tilt_deg=float(np.rad2deg(final[-1,18])),
   effective_finger_friction_min=float(min(effective_friction)) if effective_friction else -1.,
   object_mass_kg=float(mass),object_size_m=.05,z_offset_m=float(z_offset),grip_scale=float(grip_scale),
   object_initial_offset_x_m=float(offset[0]),object_initial_offset_y_m=float(offset[1]),
   grasp_bilateral_fraction_at_close=bilateral_close,grasp_confirmed=grasp_confirmed,
   drop_detected=drop_detected,drop_time_s=drop_time,max_relative_z_excess_m=float(excess.max()) if excess.size else 0.,
   contact_pairs=';'.join(' x '.join(p) for p in sorted(contact_pairs)))
 stem=f'{scenario}_{method}_{seed}'
 np.savez_compressed(out/f'{stem}.npz',data=a,columns=['t','phase','object_x','object_y','object_z','vx','vy','vz','finger_contacts','max_object_contact_force','joint_error_norm','saturation','weight_norm','finger1','finger2','relative_x','relative_y','relative_z','tilt_rad','object_qw','object_qx','object_qy','object_qz'])
 (out/f'{stem}.json').write_text(json.dumps(dict(**result,phase_peak_contact_force=phase_max),indent=2));print(json.dumps(result),flush=True)
 return result

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--batch',action='store_true');ap.add_argument('--video',action='store_true');ap.add_argument('--out',default=None);ap.add_argument('--seeds',default='31,32,33');ap.add_argument('--lift-duration',type=float,default=2.);ap.add_argument('--align-object',action='store_true');ap.add_argument('--grip-scale',type=float,default=1.);args=ap.parse_args()
 if args.batch:
  selected_path=ROOT/'results/fair_tuning_v2/selected.json'
  if not selected_path.exists():selected_path=ROOT/'results/fair_tuning/selected.json'
  selected=json.loads(selected_path.read_text());rows=[]
  out=OUT if args.out is None else Path(args.out);out.mkdir(parents=True,exist_ok=True)
  seeds=[int(s) for s in args.seeds.split(',') if s.strip()]
  for sc in ['nominal','heavy','slippery','pulse','low_friction']:
   for seed in seeds:
    for method in ['CTC','SMC','RBF_SMC']:
     rows.append(run(method,seed,sc,args.video and seed==31 and sc=='nominal' and method=='RBF_SMC',selected[method],outdir=args.out,lift_duration=args.lift_duration,align_object=args.align_object,grip_scale=args.grip_scale))
     with (out/'summary.csv').open('w') as f:
      w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 else:run(video=args.video,outdir=args.out,lift_duration=args.lift_duration,align_object=args.align_object,grip_scale=args.grip_scale)
if __name__=='__main__':main()
