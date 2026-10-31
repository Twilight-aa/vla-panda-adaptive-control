"""Replay one grasp episode and log contact geometry/forces (diagnostic only)."""
import argparse, json
from pathlib import Path
import numpy as np
import mujoco as mj
import grasp_transport as gt
from run_panda_local import ROOT, LIMIT

OUT = ROOT / 'results/grasp_diagnostics'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', default='heavy')
    ap.add_argument('--method', default='CTC')
    ap.add_argument('--seed', type=int, default=31)
    ap.add_argument('--trace-every', type=int, default=1)
    args = ap.parse_args()
    selected = json.loads((ROOT / 'results/fair_tuning_v2/selected.json').read_text())
    config = selected[args.method]
    rng = np.random.default_rng(args.seed)
    offset = np.zeros(2) if args.seed == 1 else rng.uniform(-.004, .004, 2)
    mass = .25 if args.scenario == 'heavy' else .1
    friction = .35 if args.scenario == 'slippery' else (.12 if args.scenario == 'low_friction' else 1.)
    m, d = gt.scene(mass, friction, offset)
    ctrl = gt.Controller(args.method, config)
    ref = gt.make_plan()
    obj = m.body('object').id
    hand = m.body('hand').id
    objgeom = m.geom('object_geom').id
    finger_bodies = {m.body('left_finger').id, m.body('right_finger').id}

    def label(gid):
        name = m.geom(gid).name
        return name if name else f'{m.body(m.geom_bodyid[gid]).name}:geom{gid}'

    rows = []; detail = []
    for k in range(17000):
        t = k * .001
        qd, dqd, ddqd, phase = ref(t)
        u, sat, _ = ctrl.step(d.qpos[:9].copy(), d.qvel[:9].copy(), qd, dqd, ddqd)
        d.ctrl[:7] = u
        if t < 4:
            d.ctrl[7] = 255
        elif t < 5:
            d.ctrl[7] = 255 * (1 - (t - 4))
        elif t < 12:
            d.ctrl[7] = 0
        elif t < 13:
            d.ctrl[7] = 255 * (t - 12)
        else:
            d.ctrl[7] = 255
        d.xfrc_applied[:] = 0
        if args.scenario == 'pulse' and 8 <= t < 8.15:
            d.xfrc_applied[obj, :3] = [1.5, 0, 0]
        mj.mj_step(m, d)
        obj_f = np.zeros(3); finger_f = 0.; n_finger = 0; max_force = 0.
        for ci in range(d.ncon):
            con = d.contact[ci]
            force = np.zeros(6); mj.mj_contactForce(m, d, ci, force)
            fn = float(np.linalg.norm(force[:3]))
            max_force = max(max_force, fn)
            if objgeom in (con.geom1, con.geom2):
                other = con.geom2 if con.geom1 == objgeom else con.geom1
                body = m.geom_bodyid[other]
                if body in finger_bodies:
                    n_finger += 1
                    finger_f += fn
                    detail.append(dict(
                        t=float(t), phase=int(phase), geom=label(other), body=m.body(body).name,
                        force=force[:3].round(4).tolist(), force_norm=fn,
                        pos=np.array(con.pos).round(5).tolist(), dist=float(con.dist),
                        normal=np.array(con.frame[:3]).round(4).tolist(),
                        object_pos=d.xpos[obj].round(5).tolist(),
                        object_quat=d.xquat[obj].round(5).tolist(),
                        finger_qpos=d.qpos[7:9].round(5).tolist(), ctrl=float(d.ctrl[7])))
        if k % args.trace_every == 0:
            rows.append([t, phase, *d.xpos[obj], *d.qvel[9:12], n_finger, finger_f, max_force,
                         *d.qpos[7:9], float(d.ctrl[7]), *d.xpos[hand]])
    a = np.array(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    stem = f'{args.scenario}_{args.method}_{args.seed}'
    np.savez_compressed(OUT / f'{stem}.npz', data=a, columns=[
        't', 'phase', 'obj_x', 'obj_y', 'obj_z', 'vx', 'vy', 'vz',
        'n_finger_contacts', 'finger_force_sum', 'max_contact_force',
        'finger_q1', 'finger_q2', 'gripper_ctrl', 'hand_x', 'hand_y', 'hand_z'])
    (OUT / f'{stem}_contacts.json').write_text(json.dumps(detail, indent=2))
    print(json.dumps({'scenario': args.scenario, 'method': args.method, 'seed': args.seed,
                      'contact_records': len(detail),
                      'max_finger_force': max((r['force_norm'] for r in detail), default=0.),
                      'file': str(OUT / f'{stem}.npz')}, indent=2))


if __name__ == '__main__':
    main()
