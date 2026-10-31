"""Independent gripper pull-hold test.

The arm is clamped at the task's grasp configuration.  The cube closes on the
table, then an upward force on the cube is ramped until the cube slips relative
to the fingers.  This isolates the gripper/contact capacity from arm tracking.
The measured pull capacity is the friction force the grip can sustain; the
downward-load capacity is pull capacity minus the cube weight.
"""
import argparse, csv, json
from pathlib import Path
import numpy as np
import mujoco as mj
import grasp_transport as gt
from run_panda_local import ROOT

OUT = ROOT / 'results/gripper_bench'


def make_bench(mass, friction):
    m, d = gt.scene(mass, friction, (0., 0.))
    plan = np.load(ROOT / 'results/grasp_transport/scripted_plan.npz')
    qgrasp = plan['q'][2].copy()
    d.qpos[:7] = qgrasp
    d.qpos[7:9] = .04
    mj.mj_forward(m, d)
    hand = m.body('hand').id
    rel = np.array([0., 0., .0985])
    obj_pos = d.xpos[hand] + d.xmat[hand].reshape(3, 3) @ rel
    d.qpos[9:12] = obj_pos
    d.qpos[12:16] = [1., 0., 0., 0.]
    d.qvel[:] = 0
    mj.mj_forward(m, d)
    return m, d, qgrasp


def run_once(mass, friction, force_rate=2.):
    m, d, qgrasp = make_bench(mass, friction)
    obj = m.body('object').id
    hand = m.body('hand').id
    objgeom = m.geom('object_geom').id
    finger_bodies = {m.body('left_finger').id, m.body('right_finger').id}
    dt = .001
    held0 = None
    slip_t = None; slip_force = 0.; max_normal = 0.; max_tangent = 0.
    rows = []
    for k in range(7000):
        t = k * dt
        if t < .5:
            d.ctrl[7] = 255
        elif t < 1.0:
            d.ctrl[7] = 255 * (1 - (t - .5) / .5)
        else:
            d.ctrl[7] = 0
        d.ctrl[:7] = 0
        force = min(force_rate * (t - 1.5), 12.) if t >= 1.5 else 0.
        d.xfrc_applied[:] = 0
        d.xfrc_applied[obj, 2] = force
        mj.mj_step(m, d)
        # Kinematic clamp: the arm does not move during the pull.
        d.qpos[:7] = qgrasp
        d.qvel[:7] = 0
        mj.mj_forward(m, d)
        relz = float((d.xmat[hand].reshape(3, 3).T @ (d.xpos[obj] - d.xpos[hand]))[2])
        normal_sum = 0.; tangent_sum = 0.; fingers = set()
        for ci in range(d.ncon):
            con = d.contact[ci]
            if objgeom in (con.geom1, con.geom2):
                other = con.geom2 if con.geom1 == objgeom else con.geom1
                body = m.geom_bodyid[other]
                if body in finger_bodies:
                    f = np.zeros(6); mj.mj_contactForce(m, d, ci, f)
                    normal_sum += abs(float(f[0]))
                    tangent_sum += float(np.linalg.norm(f[1:3]))
                    fingers.add(body)
        max_normal = max(max_normal, normal_sum)
        max_tangent = max(max_tangent, tangent_sum)
        if t >= 1.4 and held0 is None:
            held0 = relz
        if t >= 1.5 and slip_t is None and abs(relz - held0) > .005:
            slip_t = t
            slip_force = force
        if k % 100 == 0:
            rows.append(dict(t=round(t, 3), force_N=round(force, 3),
                             rel_z=round(relz, 5), normal_sum=round(normal_sum, 3),
                             tangent_sum=round(tangent_sum, 3), fingers=len(fingers),
                             obj_z=round(float(d.xpos[obj][2]), 4),
                             finger_q=round(float(d.qpos[7]), 5)))
    slip = slip_t is not None
    result = dict(mass_kg=mass, friction=friction,
                  held=bool(not slip),
                  initial_rel_z=float(held0),
                  slip_time_s=-1. if not slip else float(slip_t),
                  pull_capacity_N=float(slip_force if slip else 12.),
                  downward_capacity_N=float(max(0., (slip_force if slip else 12.) - mass * 9.81)),
                  max_normal_sum_N=float(max_normal), max_tangent_sum_N=float(max_tangent),
                  final_rel_z=float(relz))
    return result, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweep', action='store_true')
    ap.add_argument('--mass', type=float, default=.1)
    ap.add_argument('--friction', type=float, default=1.)
    ap.add_argument('--force-rate', type=float, default=2.)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.sweep:
        rows = []
        for friction in [1.0, .5, .35, .2, .12]:
            for mass in [.05, .1, .15, .25]:
                r, _ = run_once(mass, friction, force_rate=args.force_rate)
                rows.append(r)
                print(json.dumps(r), flush=True)
                with (OUT / 'pull_sweep.csv').open('w') as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        (OUT / 'pull_sweep.json').write_text(json.dumps(rows, indent=2))
    else:
        r, trace = run_once(args.mass, args.friction, force_rate=args.force_rate)
        with (OUT / 'pull_trace.csv').open('w') as f:
            w = csv.DictWriter(f, fieldnames=list(trace[0])); w.writeheader(); w.writerows(trace)
        print(json.dumps(r, indent=2))


if __name__ == '__main__':
    main()
