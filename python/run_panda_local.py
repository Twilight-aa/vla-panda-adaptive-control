"""Panda free-motion torque control. Independent nominal model; no VLA/grasp."""
from pathlib import Path
import argparse, csv, hashlib, json, platform, time
import numpy as np
import mujoco as mj
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'vendor/mujoco_menagerie/franka_emika_panda/scene.xml'
HOME=np.array([0.,-.25,0.,-1.7,0.,1.5,-.7853])
AMP=np.array([.12,.1,.08,.1,.08,.1,.08])
LIMIT=np.array([87,87,87,87,12,12,12.])
METHODS=['CTC','SMC','RBF_SMC','RBF_NO_ROBUST','RBF_FROZEN']

def setup(payload=0.,friction=1.):
    m=mj.MjModel.from_xml_path(str(MODEL)); m.opt.timestep=.001
    # Convert only the seven arm actuators from affine position servos to torque.
    m.actuator_gainprm[:7]=0; m.actuator_gainprm[:7,0]=1
    m.actuator_biasprm[:7]=0
    m.actuator_ctrlrange[:7]=np.column_stack((-LIMIT,LIMIT))
    m.dof_damping[:7]*=friction
    d=mj.MjData(m); mj.mj_resetDataKeyframe(m,d,0)
    if payload:
        # Added rigid mass at existing hand inertial COM; finite isotropic inertia.
        # This is not a grasped object's mesh or a simulated attachment event.
        bid=m.body('hand').id
        m.body_mass[bid]+=payload
        m.body_inertia[bid]+=2/5*payload*.04**2
        mj.mj_setConst(m,d)
        # mj_setConst resets MjData: restore the home keyframe so the gripper
        # starts open (finger qpos 0.04) in payload scenarios too. Without this
        # the fingers start closed and their pads touch at t=0 with zero force.
        mj.mj_resetDataKeyframe(m,d,0)
    return m,d

def ref(t):
    w=2*np.pi/6
    return HOME+AMP*np.sin(w*t), AMP*w*np.cos(w*t), -AMP*w*w*np.sin(w*t)

def bias(m,d,q,v):
    d.qpos[:]=q; d.qvel[:]=v; mj.mj_forward(m,d)
    return d.qfrc_bias.copy()

def run(method,scenario,seed,out,video=False,options=None):
    options={} if options is None else options
    mass=.5 if scenario in ['payload','combined','low_torque'] else (.8 if scenario=='unseen_payload' else 0.)
    m,d=setup(mass,3 if scenario in ['combined','friction'] else 1)
    nom,nd=setup(); rd=mj.MjData(nom)
    rng=np.random.default_rng(seed)
    d.qpos[:7]=HOME+rng.normal(0,options.get('initial_std',.01),7); d.qvel[:7]=ref(0)[1]
    d.ctrl[7]=255; mj.mj_forward(m,d)
    hand=m.body('hand').id
    dt=m.opt.timestep
    limits=LIMIT*(.45 if scenario=='low_torque' else 1.)
    from controllers import TorqueController
    ctrl=TorqueController(method,options,nominal=(nom,nd),dt=dt)
    logs=[]; renderer=None; writer=None
    if video:
        import imageio.v2 as iio
        renderer=mj.Renderer(m,height=480,width=640)
        camera=mj.MjvCamera(); camera.lookat[:]=[0,0,.5]; camera.distance=2.1
        camera.azimuth=130; camera.elevation=-25
        writer=iio.get_writer(out/'panda_combined.mp4',fps=25)
    start=time.perf_counter(); max_warn=0; max_contact=0; mismatch=0.
    contact_steps=0; contact_pairs=set(); max_contact_force=0.; first_contact=None
    geom_labels=[]
    for gi in range(m.ngeom):
        label=m.geom(gi).name
        geom_labels.append(label if label else f'{m.body(m.geom_bodyid[gi]).name}:geom{gi}')
    initial_contacts=int(d.ncon)
    for k in range(6000):
        t=k*dt; qd,dqd,ddqd=ref(t)
        q=d.qpos.copy(); v=d.qvel.copy()
        if options.get('noise_std',0):
            q[:7]+=rng.normal(0,options['noise_std'],7);v[:7]+=rng.normal(0,options['noise_std'],7)
        e=q[:7]-qd; de=v[:7]-dqd
        u,sat,comp=ctrl.step(q,v,qd,dqd,ddqd,limits=limits,t=t)
        d.ctrl[:7]=u
        d.xfrc_applied[:]=0
        if scenario in ['pulse','combined'] and 3<=t<3.2:
            d.xfrc_applied[hand,:3]=[8,-4,0]
        rd.qpos[:]=q; rd.qpos[:7]=qd; rd.qvel[:]=0; mj.mj_forward(nom,rd)
        ee=np.linalg.norm(d.xpos[hand]-rd.xpos[hand])
        force=d.xfrc_applied[hand,:3].copy()
        logs.append(np.r_[t,d.qpos[:7],d.qvel[:7],qd,u,d.qpos[:7]-qd,ee,np.linalg.norm(ctrl.W),comp,force,sat])
        if renderer and k%40==0:
            renderer.update_scene(d,camera); frame=renderer.render(); writer.append_data(frame)
            if k==3200:
                import imageio.v2 as iio
                iio.imwrite(out/'panda_snapshot.png',frame)
        mj.mj_step(m,d)
        if d.ncon:
            contact_steps+=1
            if first_contact is None:first_contact=t
            for ci in range(d.ncon):
                con=d.contact[ci]
                pair=tuple(sorted((geom_labels[con.geom1],geom_labels[con.geom2])))
                contact_pairs.add(pair)
                cf=np.zeros(6);mj.mj_contactForce(m,d,ci,cf)
                max_contact_force=max(max_contact_force,float(np.linalg.norm(cf[:3])))
        mismatch=max(mismatch,float(np.max(np.abs(d.qfrc_actuator[:7]-u))))
        max_warn=max(max_warn,int(sum(w.number for w in d.warning)))
        max_contact=max(max_contact,d.ncon)
        if not np.isfinite(d.qpos).all() or np.max(np.abs(d.qvel))>100:
            raise RuntimeError(f'Unstable {method}/{scenario}/{seed}')
    if writer: writer.close(); renderer.close()
    a=np.array(logs)
    header=['t']+[f'{p}{i+1}' for p in ['q','dq','qd','tau','e'] for i in range(7)]
    header+=['ee_error_m','weight_norm']+[f'comp{i+1}' for i in range(7)]+['Fx','Fy','Fz','saturation']
    if options.get('save_traces',seed==11):
        np.savez_compressed(out/f'{scenario}_{method}.npz',data=a,columns=header)
    result=dict(scenario=scenario,controller=method,seed=seed,
        joint_rmse_rad=float(np.sqrt(np.mean(np.sum(a[:,29:36]**2,axis=1)))),
        ee_rmse_m=float(np.sqrt(np.mean(a[:,36]**2))),
        late_ee_rmse_m=float(np.sqrt(np.mean(a[a[:,0]>=4.5,36]**2))),
        torque_rms_Nm=float(np.sqrt(np.mean(np.sum(a[:,22:29]**2,axis=1)))),
        max_weight_norm=float(a[:,37].max()),saturation_fraction=float(a[:,-1].mean()),
        actuator_mapping_error=mismatch,warning_count=max_warn,max_contacts=max_contact,
        initial_contacts=initial_contacts,contact_steps=contact_steps,
        contact_pairs=';'.join(' x '.join(p) for p in sorted(contact_pairs)),
        max_contact_force_N=max_contact_force,first_contact_time_s=-1. if first_contact is None else first_contact,
        wall_seconds=time.perf_counter()-start)
    return result

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--smoke',action='store_true'); ap.add_argument('--video',action='store_true'); ap.add_argument('--cartesian',action='store_true'); args=ap.parse_args()
    if args.cartesian:
        from scipy.interpolate import CubicSpline
        global ref
        stored=np.load(ROOT/'results/cartesian_reference/reference.npz')
        spline=CubicSpline(stored['t'],stored['q'],bc_type='natural')
        ref=lambda t: (spline(t),spline(t,1),spline(t,2))
    out=ROOT/'results'/('panda_smoke' if args.smoke else ('panda_cartesian' if args.cartesian else 'panda_local')); out.mkdir(parents=True,exist_ok=True)
    rows=[]
    scenarios=['nominal'] if args.smoke else ['nominal','payload','pulse','combined']
    seeds=[11] if args.smoke else [11,12,13]
    methods=['CTC'] if args.smoke else METHODS
    for seed in seeds:
        for sc in scenarios:
            for method in methods:
                result=run(method,sc,seed,out,args.video and seed==11 and sc=='combined' and method=='RBF_SMC')
                rows.append(result); print(json.dumps(result),flush=True)
                with (out/'summary.csv').open('w') as f:
                    wr=csv.DictWriter(f,fieldnames=list(rows[0])); wr.writeheader();wr.writerows(rows)
    meta={'mujoco':mj.__version__,'python':platform.python_version(),'platform':platform.platform(),
          'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'cartesian_reference':args.cartesian,'note':'Local Menagerie Panda, not LIBERO; offline reference, no grasp or VLA.'}
    (out/'metadata.json').write_text(json.dumps(meta,indent=2))
    if not args.smoke:
        fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained')
        for axis,sc in zip(ax.flat,scenarios):
            for method in methods:
                a=np.load(out/f'{sc}_{method}.npz')['data']; axis.plot(a[:,0],a[:,36]*1000,label=method)
            axis.set(title=sc,xlabel='Time (s)',ylabel='Hand-frame position error (mm)');axis.grid();axis.legend(fontsize=7)
        fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
if __name__=='__main__': main()
