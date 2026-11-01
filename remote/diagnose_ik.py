"""Diagnose the damped-least-squares IK used by libero_torque_wrapper.

Run from /root/openvla-oft/experiments/robot/libero after the checkout is on
PYTHONPATH.  It does not load the VLA model; it only creates the LIBERO env,
advances the arm with a fixed action, and reports IK residuals for several
iteration/damping/step settings.
"""
import numpy as np
import mujoco
from robosuite.utils.control_utils import orientation_error

import run_libero_eval as rle
from libero.libero import benchmark
from libero_torque_wrapper import install


def solve_ik(c, q0, target_pos, target_ori, iters=40, damping=1e-3, step=.5, use_ori=True):
    q = q0.copy()
    lo = c.m.jnt_range[c.qidx, 0] + .01
    hi = c.m.jnt_range[c.qidx, 1] - .01
    d = mujoco.MjData(c.m)
    d.qpos[:] = c.sim.data._data.qpos
    d.qvel[:] = 0
    jp = np.zeros((3, c.m.nv))
    jr = np.zeros((3, c.m.nv))
    last = None
    for i in range(iters):
        d.qpos[c.qidx] = q
        mujoco.mj_forward(c.m, d)
        dp = np.asarray(target_pos) - d.site_xpos[c.site_id]
        do = orientation_error(np.asarray(target_ori), d.site_xmat[c.site_id].reshape(3, 3))
        err = np.r_[dp, do if use_ori else np.zeros(3)]
        last = (i + 1, float(np.linalg.norm(dp)), float(np.linalg.norm(do)), q.copy())
        if np.linalg.norm(err) < 1e-5:
            break
        mujoco.mj_jacSite(c.m, d, jp, jr, c.site_id)
        J = np.vstack([jp[:, c.vidx], jr[:, c.vidx]])
        dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), err)
        q = np.clip(q + step * dq, lo, hi)
    return q, last


def main():
    print("diagnose: creating benchmark", flush=True)
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(0)
    print("diagnose: creating env", flush=True)
    env, _ = rle.get_libero_env(task, "openvla", 256)
    install(env, "CTC", {"ctc_kp": 900., "ctc_kd": 66., "ref_tau": .02})
    print("diagnose: reset", flush=True)
    env.reset()
    print("diagnose: stepping", flush=True)
    action = np.array([.5, 0., 0., 0., 0., 0., -1.])
    for _ in range(60):
        env.step(action)
    print("diagnose: stepped", flush=True)
    c = env.robots[0].controller
    q0 = c.sim.data.qpos[c.qidx].copy()
    target_pos = np.asarray(c.osc.goal_pos).copy()
    target_ori = np.asarray(c.osc.goal_ori).copy()
    print("q0", np.round(q0, 4))
    print("target", np.round(target_pos, 4), "ori_shape", target_ori.shape)
    print("site", np.round(c.sim.data.site_xpos[c.site_id], 4))
    print("limits", np.round(c.m.jnt_range[c.qidx], 4))
    qw = c._ik(q0, target_pos, target_ori)
    dw = mujoco.MjData(c.m)
    dw.qpos[:] = c.sim.data._data.qpos
    dw.qpos[c.qidx] = qw
    mujoco.mj_forward(c.m, dw)
    print(
        "wrapper_ik |dp|={:.6f} |do|={:.6f} q={}".format(
            float(np.linalg.norm(target_pos - dw.site_xpos[c.site_id])),
            float(np.linalg.norm(orientation_error(target_ori, dw.site_xmat[c.site_id].reshape(3, 3)))),
            np.round(qw, 4),
        )
    )
    for use_ori in [True, False]:
        for iters in [40, 100, 300]:
            for damping in [1e-3, 1e-2, 1e-1]:
                for step in [.5, 1.0]:
                    q, last = solve_ik(c, q0, target_pos, target_ori, iters, damping, step, use_ori)
                    d = mujoco.MjData(c.m)
                    d.qpos[:] = c.sim.data._data.qpos
                    d.qpos[c.qidx] = q
                    mujoco.mj_forward(c.m, d)
                    dp = np.linalg.norm(target_pos - d.site_xpos[c.site_id])
                    do = np.linalg.norm(orientation_error(target_ori, d.site_xmat[c.site_id].reshape(3, 3)))
                    print(
                        "ori={} iters={} damp={} step={} -> |dp|={:.6f} |do|={:.6f} q={}".format(
                            int(use_ori), iters, damping, step, dp, do, np.round(q, 4)
                        )
                    )


if __name__ == "__main__":
    main()
