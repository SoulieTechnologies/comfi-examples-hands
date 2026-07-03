#!/usr/bin/env python3
"""LIVE ACADOS IK — receive gravity-aligned Goliath-70 3D over TCP from the SAM3D
streamer (stream_demo.py --emit-port) and retarget onto the MANO body in meshcat,
in real time.

Run (comfi env, server):
  source /home/users/theo/miniconda3/etc/profile.d/conda.sh && conda activate acados
  cd /home/users/theo/code/comfi-examples_new
  export ACADOS_SOURCE_DIR=$PWD/acados LD_LIBRARY_PATH=$PWD/acados/lib:$LD_LIBRARY_PATH
  export ACADOS_EXT_FUN_COMPILE_FLAGS=-O1
  python scripts/run_ik_live.py --host localhost --emit-port 8090 \
      --N 10 --w-dq 1e-3 --w-u 1e-4 --subject-height 1.75 --subject-weight 70
Then view meshcat (browser): http://<server>:7000/static/  (ssh -L 7000:localhost:7000)

The 3D received is ALREADY gravity-aligned (ROS frame) → no cv_to_ros here.
"""

import argparse
import socket
import struct
import time
from collections import deque
from pathlib import Path

import numpy as np
import pinocchio as pin

# Reuse everything from the (renamed) modified offline script, so it doesn't clash
# with your local run_ik_acados_mpc_sam3d.py.
from run_ik_acados_mpc_sam3d_live import (
    GOLIATH_NAMES, HAND_KP_TO_MANO_LINK, HAND_SOLVER_KEYS, SOLVER_KEYS,
    JOINTS_TO_LOCK, AcadosMPCIKSolver, register_tracking_frames,
    scale_model_from_goliath,
)

_MSG = 4 + 70 * 3 * 4          # 4-byte counter + 70*3 float32
_RX = {"kp": None, "n": 0}


def _rx_thread(sock):
    """Continuously read (70,3) frames; always keep only the LATEST (drop-old)."""
    buf = b""
    while True:
        try:
            d = sock.recv(65536)
        except OSError:
            return
        if not d:
            return
        buf += d
        msg = None
        while len(buf) >= _MSG:          # drain to the most recent complete message
            msg = buf[:_MSG]
            buf = buf[_MSG:]
        if msg is not None:
            _RX["n"] = struct.unpack(">I", msg[:4])[0]
            _RX["kp"] = np.frombuffer(msg[4:], dtype=np.float32).reshape(70, 3).copy()


def _frame_dict(kp):
    return {name: kp[i] for i, name in enumerate(GOLIATH_NAMES)}


def main():
    p = argparse.ArgumentParser(description="Live ACADOS IK from streamed SAM3D 3D")
    p.add_argument("--host", default="localhost", help="stream_demo.py host")
    p.add_argument("--emit-port", type=int, default=8090)
    p.add_argument("--model-dir", default=str(Path(__file__).resolve().parent.parent / "model"))
    p.add_argument("--urdf-file", default="urdf/human_with_mano.urdf")
    p.add_argument("--N", type=int, default=10)
    p.add_argument("--dt", type=float, default=0.033)
    p.add_argument("--w-markers", type=float, default=1.0)
    p.add_argument("--w-hand", type=float, default=3.0)
    p.add_argument("--w-fingertip-mult", type=float, default=2.0)
    p.add_argument("--w-dq", type=float, default=1e-3)
    p.add_argument("--w-dq-hand", type=float, default=None)
    p.add_argument("--w-u", type=float, default=1e-4)
    p.add_argument("--subject-height", type=float, default=1.75)
    p.add_argument("--subject-weight", type=float, default=70.0)
    p.add_argument("--gender", default="male", choices=["male", "female"])
    p.add_argument("--augmenter-path", default=str(Path(__file__).resolve().parent.parent / "augmentation_model"))
    p.add_argument("--warmup", type=int, default=30, help="frames to collect before starting")
    args = p.parse_args()

    urdf_path = str(Path(args.model_dir) / args.urdf_file)

    # ── connect + collect warmup frames ──────────────────────────────────────
    print(f"[1/5] Connecting to {args.host}:{args.emit_port} ...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((args.host, args.emit_port))
    import threading
    threading.Thread(target=_rx_thread, args=(sock,), daemon=True).start()

    print(f"[2/5] Collecting {args.warmup} warmup frames...")
    warm = []
    last = -1
    while len(warm) < args.warmup:
        if _RX["kp"] is not None and _RX["n"] != last:
            last = _RX["n"]
            warm.append(_RX["kp"].copy())
        else:
            time.sleep(0.01)
    warm_arr = np.stack(warm)

    # ── model: load, scale (COSMIK+LSTM), register frames, lock thorax ────────
    print("[3/5] Building + scaling model...")
    human_model, human_coll, human_vis = pin.buildModelsFromUrdf(
        urdf_path, str(Path(urdf_path).parent), root_joint=pin.JointModelFreeFlyer())
    human_model = scale_model_from_goliath(
        human_model, warm_arr, augmenter_path=args.augmenter_path,
        subject_height=args.subject_height, subject_weight=args.subject_weight,
        gender=args.gender)
    human_model, registered = register_tracking_frames(human_model)
    lock = [human_model.getJointId(j) for j in JOINTS_TO_LOCK if human_model.existJointName(j)]
    q0 = pin.neutral(human_model)
    if lock:
        human_model, (human_coll, human_vis) = pin.buildReducedModel(
            human_model, [human_coll, human_vis], lock, q0)
    human_data = pin.Data(human_model)
    valid_keys = [k for k in SOLVER_KEYS if human_model.existFrame(k)]

    # ── meshcat ──────────────────────────────────────────────────────────────
    print("[4/5] Meshcat...")
    import meshcat.geometry as mg
    import meshcat.transformations as mtf
    from pinocchio.visualize import MeshcatVisualizer
    viz = MeshcatVisualizer(human_model, human_coll, human_vis)
    viz.initViewer(open=False)
    viz.loadViewerModel("human")
    viewer = viz.viewer
    for mk in valid_keys:
        viewer[f"gt/{mk}"].set_object(mg.Sphere(0.012), mg.MeshLambertMaterial(color=0xFF0000, opacity=0.6))
    viz.display(pin.neutral(human_model))
    print(f"      open meshcat: {viewer.url()}  (ssh -L 7000:localhost:7000)")

    # ── IPOPT warm-start + acados solver ─────────────────────────────────────
    print("[5/5] IPOPT warm-start + acados build (~1 min first time)...")
    from comfi_examples.ik_utils import RT_IK
    wkeys = [k for k in registered if human_model.existFrame(k)]
    omega = {k: (0.5 if k in HAND_KP_TO_MANO_LINK else 1.0) for k in wkeys}
    fd0 = _frame_dict(warm_arr[-1])
    q_init = RT_IK(human_model, fd0, pin.neutral(human_model), wkeys, args.dt, omega).solve_ik_sample_casadi()
    ik = AcadosMPCIKSolver(
        pin_model=human_model, keys_to_track=valid_keys, N=args.N, dt=args.dt,
        w_markers=args.w_markers, w_hand=args.w_hand, w_dq=args.w_dq,
        w_dq_hand=args.w_dq_hand, w_u=args.w_u, hand_keys=HAND_SOLVER_KEYS,
        w_fingertip_mult=args.w_fingertip_mult)
    ik.reset_warm_start(q_init)

    window = deque([_frame_dict(f) for f in warm_arr[-args.N:]], maxlen=args.N)
    while len(window) < args.N:
        window.appendleft(window[0])

    print("LIVE — retargeting... (Ctrl+C to stop)")
    last = -1
    solved = 0
    t_report = time.time()
    while True:
        if _RX["kp"] is None or _RX["n"] == last:
            time.sleep(0.002)
            continue
        last = _RX["n"]
        window.append(_frame_dict(_RX["kp"]))
        q = ik.solve(list(window))
        viz.display(q)
        pin.forwardKinematics(human_model, human_data, q)
        pin.updateFramePlacements(human_model, human_data)
        cur = window[-1]
        for mk in valid_keys:
            gt = np.asarray(cur[mk]).flatten()[:3]
            if np.isfinite(gt).all():
                viewer[f"gt/{mk}"].set_transform(mtf.translation_matrix(gt))
        solved += 1
        if time.time() - t_report > 2.0:
            print(f"  IK {solved / (time.time() - t_report):.1f} solves/s", flush=True)
            solved = 0
            t_report = time.time()


if __name__ == "__main__":
    main()
