#!/usr/bin/env python3

"""Sliding-window MPC-style inverse kinematics solver using acados.

The IK is formulated as a multi-stage OCP over a horizon of N frames:
State: x = [q; dq] (nq + nv)
Control: u = ddq (nv)
Dynamics: q_next = integrate(q, dq * dt)
dq_next = dq + ddq * dt

For each output frame i (causal, uses past N frames):
- Use a window of N past marker frames [p[i-N+1], ..., p[i]]
- Stage k cost (k=0..N-1):
  w_markers * ||markers(q_k) - p[i-N+1+k]||²
  w_dq * ||dq_k||²
  w_u * ||u_k||²
- Terminal cost: w_dq * ||dq_N||² (no marker target beyond window)
- Output: q_{N-1} (last stage = current frame i)
- Advance window by 1 and shift warm-start

Requires: pinocchio, pinocchio.casadi (cpin), casadi, acados_template
"""

import os
import sys
import argparse
import time
from pathlib import Path

import casadi
import imageio
import meshcat
import meshcat.geometry
import meshcat.transformations
import numpy as np
import pandas as pd
import pinocchio as pin
import pinocchio.casadi as cpin
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from pinocchio.visualize import MeshcatVisualizer

# ── Constants ─────────────────────────────────────────────────────────────

SUBJECT_IDS = [
    "1012", "1118", "1508", "1602", "1847", "2112", "2198", "2307",
    "3361", "4162", "4216", "4279", "4509", "4612", "4665", "4687",
    "4801", "4827",
]

TASKS = [
    "Screwing", "ScrewingSat", "Crouching", "Picking", "Hammering",
    "HammeringSat", "Jumping", "Lifting", "QuickLifting", "Lower",
    "SideOverhead", "FrontOverhead", "RobotPolishing", "RobotWelding",
    "Polishing", "PolishingSat", "SitToStand", "Squatting", "Static",
    "Upper", "CircularWalking", "StraightWalking", "Welding", "WeldingSat",
]

# ── THE ROSETTA STONE ──
# This list must perfectly match the index order of axis 1 in your .npy file.
UNIFIED_MARKERS = [
    # --- SMPL BODY (Indices 0-23) ---
    "pelvis",           # 0
    "left_hip",         # 1
    "right_hip",        # 2
    "spine1",           # 3
    "left_knee",        # 4
    "right_knee",       # 5
    "spine2",           # 6
    "left_ankle",       # 7
    "right_ankle",      # 8
    "spine3",           # 9
    "left_foot",        # 10
    "right_foot",       # 11
    "neck",             # 12
    "left_collar",      # 13
    "right_collar",     # 14
    "head",             # 15
    "left_shoulder",    # 16
    "right_shoulder",   # 17
    "left_elbow",       # 18
    "right_elbow",      # 19
    "left_wrist",       # 20
    "right_wrist",      # 21
    "left_hand_dummy",  # 22 (SMPL hand base, safely ignored if not in URDF)
    "right_hand_dummy", # 23 (SMPL hand base, safely ignored if not in URDF)

    # --- FACE / EXTRA (Indices 24-27) ---
    "nose",             # 24 
    "right_eye",        # 25
    "left_eye",         # 26
    "right_ear",        # 27

    # --- LEFT HAND (Indices 28-48) ---
    "left_thumb_fingertip", "left_thumb_DP", "left_thumb_PP", "left_thumb_MC",
    "left_index_fingertip", "left_index_DP", "left_index_MP", "left_index_PP",
    "left_middle_fingertip", "left_middle_DP", "left_middle_MP", "left_middle_PP",
    "left_ring_fingertip", "left_ring_DP", "left_ring_MP", "left_ring_PP",
    "left_pinky_fingertip", "left_pinky_DP", "left_pinky_MP", "left_pinky_PP",
    "left_hand_wrist",  # 48: Left Wrist

    # --- RIGHT HAND (Indices 49-69) ---
    "right_thumb_fingertip", "right_thumb_DP", "right_thumb_PP", "right_thumb_MC",
    "right_index_fingertip", "right_index_DP", "right_index_MP", "right_index_PP",
    "right_middle_fingertip", "right_middle_DP", "right_middle_MP", "right_middle_PP",
    "right_ring_fingertip", "right_ring_DP", "right_ring_MP", "right_ring_PP",
    "right_pinky_fingertip", "right_pinky_DP", "right_pinky_MP", "right_pinky_PP",
    "right_hand_wrist"  # 69: Right Wrist
]

SOLVER_KEYS = [
    "pelvis", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_thumb_fingertip", "left_index_fingertip", "left_middle_fingertip", "left_ring_fingertip", "left_pinky_fingertip",
    "right_thumb_fingertip", "right_index_fingertip", "right_middle_fingertip", "right_ring_fingertip", "right_pinky_fingertip"
]

# ── Acados MPC IK Solver ──────────────────────────────────────────────────

class AcadosMPCIKSolver:
    def __init__(
        self,
        pin_model: pin.Model,
        keys_to_track: list,
        N: int = 10,
        dt: float = 0.025,
        w_markers: float = 1.0,
        w_dq: float = 1e-3,
        w_u: float = 1e-4,
    ):
        self._pin_model = pin_model
        self._nq = pin_model.nq
        self._nv = pin_model.nv
        self._nx = self._nq + self._nv
        self._nu = self._nv
        self._N = N
        self._dt = dt
        self._keys_to_track = keys_to_track
        
        # Count how many markers actually exist in the URDF model
        self._valid_keys = [k for k in keys_to_track if self._pin_model.existFrame(k)]
        self._n_markers = len(self._valid_keys)
        self._nmc = 3 * self._n_markers

        self._w_markers = w_markers
        self._w_dq = w_dq
        self._w_u = w_u

        self._cmodel = cpin.Model(self._pin_model)
        self._cdata = self._cmodel.createData()
        self._ocp_solver = self._create_ocp_solver()

    def _build_marker_fk_expr(self, cq):
        cpin.framesForwardKinematics(self._cmodel, self._cdata, cq)
        markers_expr = []
        for key in self._valid_keys:
            frame_id = self._cmodel.getFrameId(key)
            markers_expr.append(self._cdata.oMf[frame_id].translation)
        return casadi.vertcat(*markers_expr)

    def _create_ocp_solver(self):
        nmc = self._nmc
        cx = casadi.SX.sym("x", self._nx)
        cu = casadi.SX.sym("u", self._nu)
        cq = cx[: self._nq]
        cdq = cx[self._nq:]

        q_next = cpin.integrate(self._cmodel, cq, cdq * self._dt)
        dq_next = cdq + cu * self._dt
        x_next = casadi.vertcat(q_next, dq_next)
        markers_expr = self._build_marker_fk_expr(cq)

        model = AcadosModel()
        model.name = f"human_ik_mpc_N{self._N}"
        model.x = cx
        model.u = cu
        model.disc_dyn_expr = x_next
        model.cost_y_expr = casadi.vertcat(markers_expr, cdq, cu)
        model.cost_y_expr_e = cdq
        p = casadi.SX.sym("p", nmc)
        model.p = p

        ocp = AcadosOcp()
        ocp.model = model
        ocp.solver_options.N_horizon = self._N
        ocp.solver_options.tf = self._N * self._dt
        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.cost_type_e = "NONLINEAR_LS"

        ny = nmc + self._nv + self._nu
        ocp.cost.yref = np.zeros(ny)
        W = np.zeros((ny, ny))
        W[:nmc, :nmc] = self._w_markers * np.eye(nmc)
        W[nmc:nmc + self._nv, nmc:nmc + self._nv] = self._w_dq * np.eye(self._nv)
        W[nmc + self._nv:, nmc + self._nv:] = self._w_u * np.eye(self._nu)
        ocp.cost.W = W

        ny_e = self._nv
        ocp.cost.yref_e = np.zeros(ny_e)
        ocp.cost.W_e = self._w_dq * np.eye(ny_e)

        # FreeFlyer root uses first 7 dims [x, y, z, qx, qy, qz, qw]
        n_joint_constraints = self._nq - 7
        if n_joint_constraints > 0:
            q_constrained = cx[7:self._nq]
            model.con_h_expr = q_constrained
            lh = np.array(self._pin_model.lowerPositionLimit[7:self._nq])
            uh = np.array(self._pin_model.upperPositionLimit[7:self._nq])
            ocp.constraints.lh = lh
            ocp.constraints.uh = uh

        ocp.constraints.x0 = np.zeros(self._nx)
        ocp.parameter_values = np.zeros(nmc)

        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "DISCRETE"
        ocp.solver_options.nlp_solver_type = "SQP"
        ocp.solver_options.nlp_solver_max_iter = 50
        ocp.solver_options.qp_solver_iter_max = 100
        ocp.solver_options.tol = 1e-4
        ocp.solver_options.globalization = "MERIT_BACKTRACKING"

        os.environ["ACADOS_SOURCE_DIR"] = str(Path(__file__).resolve().parent.parent / "acados")
        return AcadosOcpSolver(ocp)

    def reset_warm_start(self, q_init: np.ndarray):
        x0 = np.zeros(self._nx)
        x0[:self._nq] = q_init
        for k in range(self._N + 1):
            self._ocp_solver.set(k, "x", x0)
        for k in range(self._N):
            self._ocp_solver.set(k, "u", np.zeros(self._nu))

    def solve(self, marker_window: list) -> np.ndarray:
        x0_warm = self._ocp_solver.get(0, "x")
        self._ocp_solver.constraints_set(0, "lbx", x0_warm)
        self._ocp_solver.constraints_set(0, "ubx", x0_warm)

        for k in range(self._N):
            p_meas = np.zeros(self._nmc)
            for i, key in enumerate(self._valid_keys):
                p_meas[3*i : 3*i+3] = np.array(marker_window[k][key]).flatten()

            yref_k = np.concatenate([p_meas, np.zeros(self._nv), np.zeros(self._nu)])
            self._ocp_solver.set(k, "yref", yref_k)
            self._ocp_solver.set(k, "p", p_meas)

        self._ocp_solver.set(self._N, "yref", np.zeros(self._nv))
        self._ocp_solver.solve()

        x_sol = self._ocp_solver.get(self._N - 1, "x")
        q_sol = x_sol[:self._nq]

        for k in range(self._N):
            x_k_next = self._ocp_solver.get(k + 1, "x")
            self._ocp_solver.set(k, "x", x_k_next)
            if k < self._N - 1:
                u_k_next = self._ocp_solver.get(k + 1, "u")
                self._ocp_solver.set(k, "u", u_k_next)

        return q_sol


# ── Model loading ──────────────────────────────

def load_unified_model(urdf_path: str, model_dir: str):
    """Load the full body + hands URDF with a Freeflyer root."""
    print(f"[INFO] Loading Unified URDF from {urdf_path}")
    
    root_joint = pin.JointModelFreeFlyer()
    hand_package_dir = os.path.join(model_dir, "sharpa_hand", "wave_01")
    package_dirs = [model_dir, hand_package_dir]
    model, collision_model, visual_model = pin.buildModelsFromUrdf(
        urdf_path, 
        root_joint=root_joint, 
        package_dirs=package_dirs
    )
    print(f"[INFO] Unified Model DOF: {model.nq} (including 7 for FreeFlyer)")
    return model, collision_model, visual_model


# ── Main ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Sliding-window MPC IK solver using acados")
    p.add_argument("--id", dest="subject_id", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--comfi-root", default=Path(os.environ.get("COMFI_ROOT", "COMFI")), type=Path)
    p.add_argument("--output-root", default=Path("output").resolve() / "res_hpe", type=Path)
    p.add_argument("--model-dir", default=Path("model"), type=Path)
    p.add_argument("--urdf-file", default="urdf/human_with_sharpa.urdf")
    p.add_argument("--keypoints-file", default="dexsuite_joints_sam3d.npy")
    p.add_argument("--start-sample", type=int, default=0)
    p.add_argument("--dt", type=float, default=0.025)
    p.add_argument("--N", type=int, default=3)
    p.add_argument("--w-markers", type=float, default=1.0)
    p.add_argument("--w-dq", type=float, default=1e-5)
    p.add_argument("--w-u", type=float, default=1e-6)
    p.add_argument("--display", action="store_true")
    p.add_argument("--save-video", action="store_true")
    p.add_argument("--meshcat-url", default="tcp://127.0.0.1:6000")
    p.add_argument("--output-angles-file", default="joint_angles_acados_mpc.csv")
    p.add_argument("--output-markers-file", default="markers_model_trajectories_acados_mpc.csv")
    return p.parse_args()


def main():
    args = parse_args()

    output_root = Path(args.output_root).resolve()
    model_dir = Path(args.model_dir).resolve()

    path_to_kpt = output_root / args.subject_id / args.task / args.keypoints_file
    output_markers_csv = output_root / args.subject_id / args.task / args.output_markers_file
    output_angles_csv = output_root / args.subject_id / args.task / args.output_angles_file
    merged_urdf = str(model_dir / args.urdf_file)

    if not path_to_kpt.exists():
        raise FileNotFoundError(f"Missing target Numpy data: {path_to_kpt}")
    if not Path(merged_urdf).exists():
        raise FileNotFoundError(f"Missing URDF: {merged_urdf}")

    # 1. Load the Unified Model (FreeFlyer)
    human_model, human_collision_model, human_visual_model = load_unified_model(merged_urdf, str(model_dir))
    human_data = pin.Data(human_model)

    # 2. Load and Format the 70-point Data Tensor
    print(f"[INFO] Loading unified tracking data from {path_to_kpt}...")
    tracking_data = np.load(str(path_to_kpt)) # Expected shape: (N, 70, 3)
    n_frames = tracking_data.shape[0]
    
    if tracking_data.shape[1] != len(UNIFIED_MARKERS):
        raise ValueError(f"Data has {tracking_data.shape[1]} joints, but UNIFIED_MARKERS lists {len(UNIFIED_MARKERS)}.")

    # Convert OpenCV frame (Right, Down, Forward) to ROS frame (Forward, Left, Up)
    cv_x = tracking_data[:, :, 0].copy()
    cv_y = tracking_data[:, :, 1].copy()
    cv_z = tracking_data[:, :, 2].copy()
    
    tracking_data[:, :, 0] = cv_z   # X = Forward
    tracking_data[:, :, 1] = -cv_x  # Y = Left
    tracking_data[:, :, 2] = -cv_y  # Z = Up
    
    result_markers = []
    for frame_idx in range(n_frames):
        frame_dict = {}
        for kp_idx, frame_name in enumerate(UNIFIED_MARKERS):
            frame_dict[frame_name] = tracking_data[frame_idx, kp_idx, :]
        result_markers.append(frame_dict)

    print(f"[INFO] Prepared {n_frames} full-body frames for IK tracking")

    # ── Visualization setup ──
    viz = None
    viewer = None
    if args.display:
        viz = MeshcatVisualizer(human_model, human_collision_model, human_visual_model)
        viewer = meshcat.Visualizer()
        print(f"[INFO] Meshcat web URL: {viewer.url()}")
        viz.initViewer(viewer=viewer, open=True)
        viz.viewer.delete()
        viz.loadViewerModel("human_model")

        viewer["/Background"].set_property("top_color", [1, 1, 1])
        viewer["/Background"].set_property("bottom_color", [0.65, 0.65, 0.65])

        for marker_name in UNIFIED_MARKERS:
            # Only visualize markers that actually map to links in the URDF
            if human_model.existFrame(marker_name):
                viewer[f"gt_markers/{marker_name}"].set_object(
                    meshcat.geometry.Sphere(0.015),
                    meshcat.geometry.MeshLambertMaterial(color=0xFF0000, opacity=0.6),
                )
                viewer[f"model_markers/{marker_name}"].set_object(
                    meshcat.geometry.Sphere(0.01),
                    meshcat.geometry.MeshLambertMaterial(color=0x000000, opacity=0.6),
                )

        q0 = pin.neutral(human_model)
        viz.display(q0)
        first_mks = result_markers[max(0, args.start_sample)]
        for marker_name in UNIFIED_MARKERS:
            if human_model.existFrame(marker_name) and marker_name in first_mks:
                pos = np.array(first_mks[marker_name]).flatten()
                viewer[f"gt_markers/{marker_name}"].set_transform(
                    meshcat.transformations.translation_matrix(pos)
                )

        print("[INFO] Visualization ready — press Enter to start MPC IK solving...")
        input()

    # Create MPC IK solver
    print(f"[INFO] Creating acados MPC IK solver (N={args.N}, dt={args.dt})...")
    ik_solver = AcadosMPCIKSolver(
        pin_model=human_model,
        keys_to_track=SOLVER_KEYS, 
        N=args.N,
        dt=args.dt,
        w_markers=args.w_markers,
        w_dq=args.w_dq,
        w_u=args.w_u,
    )

    # Solve first frame using IPOPT so the FreeFlyer starts in the right spot!
    print("[INFO] Warm-starting first-frame IK with IPOPT...")
    from comfi_examples.ik_utils import RT_IK
    
    # Filter omega weights so RT_IK doesn't crash searching for dummy facial markers
    valid_omega = {k: 1.0 for k in UNIFIED_MARKERS if human_model.existFrame(k)}
    q0 = pin.neutral(human_model)
    rt_ik = RT_IK(
        human_model, result_markers[args.start_sample], q0,
        list(valid_omega.keys()), args.dt, valid_omega,
    )
    q_init = rt_ik.solve_ik_sample_casadi()
    ik_solver.reset_warm_start(q_init)

    print(f"[INFO] Solving MPC IK for frames [{args.start_sample}, {n_frames})...")

    q_list = []
    M_model_list = []
    rmse_per_marker = {}
    solve_times = []
    video_frames = [] if (args.display and args.save_video) else None

    for ii in range(args.start_sample, n_frames):
        marker_window = [
            result_markers[max(args.start_sample, ii - args.N + 1 + k)]
            for k in range(args.N)
        ]
        current_mks = marker_window[-1] 

        t0 = time.perf_counter()
        q = ik_solver.solve(marker_window)
        solve_times.append(time.perf_counter() - t0)

        pin.forwardKinematics(human_model, human_data, q)
        pin.updateFramePlacements(human_model, human_data)

        M_model_frame = {}
        for marker in current_mks.keys():
            if not human_model.existFrame(marker):
                continue
                
            pos_gt = np.array(current_mks[marker]).flatten()
            M_model_mat = human_data.oMf[human_model.getFrameId(marker)]
            pos_model = np.array(M_model_mat.translation).flatten()

            M_model_frame[f"{marker}_x"] = M_model_mat.translation[0]
            M_model_frame[f"{marker}_y"] = M_model_mat.translation[1]
            M_model_frame[f"{marker}_z"] = M_model_mat.translation[2]

            if viewer:
                viewer[f"gt_markers/{marker}"].set_transform(
                    meshcat.transformations.translation_matrix(pos_gt)
                )
                viewer[f"model_markers/{marker}"].set_transform(
                    meshcat.transformations.translation_matrix(pos_model)
                )

            sq_error = np.sum((pos_gt - pos_model) ** 2)
            rmse_per_marker.setdefault(marker, []).append(sq_error)

        M_model_list.append(M_model_frame)
        q_list.append(q)

        if viz:
            viz.display(q)

        if video_frames is not None:
            img = viz.captureImage()
            video_frames.append(img)

        if ii % 10 == 0:
            print(f"[PROGRESS] Frame {ii}/{n_frames}")

    print("[INFO] MPC IK calculations complete")

    if video_frames is not None and len(video_frames) > 0:
        video_path = output_root / args.subject_id / args.task / "ik_acados_mpc.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        fps = int(1.0 / args.dt)
        imageio.mimsave(str(video_path), video_frames, fps=fps)

    solve_times_arr = np.array(solve_times)
    output_times_csv = output_angles_csv.parent / "solve_times_acados_mpc.csv"
    output_times_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "frame": np.arange(args.start_sample, args.start_sample + len(solve_times_arr)),
        "solve_time_s": solve_times_arr,
    }).to_csv(output_times_csv, index=False)

    pd.DataFrame(M_model_list).to_csv(output_markers_csv, index=False)
    pd.DataFrame(q_list).to_csv(output_angles_csv, index=False)

    print(f"[SUCCESS] {args.subject_id}/{args.task} acados MPC IK processing complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())