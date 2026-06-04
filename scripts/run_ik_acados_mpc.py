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

from comfi_examples.urdf_utils import Robot, mks_registration, scale_human_model
from comfi_examples.utils import read_mks_data, read_subject_yaml

# ── Constants (same as run_ik_acados.py) ──────────────────────────────────

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

MKS_TO_SKIP = [
    "LForearm", "LUArm", "RUArm", "RHJC_study", "LHJC_study",
    "r_pelvis", "l_pelvis", "LHand", "LHL2", "LHM5", "RForearm",
    "RHand", "RHL2", "RHM5", "L_sh1_study", "L_thigh1_study",
    "r_sh1_study", "r_thigh1_study", "r_thigh2_study", "L_sh2_study",
    "L_thigh2_study", "r_sh2_study", "L_sh3_study", "L_thigh3_study",
    "r_sh3_study", "r_thigh3_study",
]

KEYS_TO_ADD = ["Nose", "Head", "Right_Ear", "Left_Ear", "Right_Eye", "Left_Eye"]

KEYS_TO_TRACK = [
    "Nose", "Head", "Right_Ear", "Left_Ear", "Right_Eye", "Left_Eye",
    "C7", "RASI", "LASI", "RPSI", "LPSI", "RSHO", "RELB", "RMELB",
    "RWRI", "RMWRI", "RANK", "RMANK", "RTOE", "R5MHD", "RHEE", "RKNE",
    "RMKNE", "LSHO", "LELB", "LMELB", "LWRI", "LMWRI", "LANK", "LMANK",
    "LTOE", "L5MHD", "LHEE", "LKNE", "LMKNE",
]

JOINTS_TO_LOCK = [
    "middle_thoracic_X", "middle_thoracic_Y", "middle_thoracic_Z",
    "left_wrist_X", "left_wrist_Z", "right_wrist_X", "right_wrist_Z",
]

JOINT_ANGLES_NAMES = [
    "Freeflyer_X[m]", "Freeflyer_Y[m]", "Freeflyer_Z[m]",
    "Freeflyer_quaternion_X", "Freeflyer_quaternion_Y",
    "Freeflyer_quaternion_Z", "Freeflyer_quaternion_W",
    "Left_Hip_Flexion_Extension[rad]", "Left_Hip_Abduction_Adduction[rad]",
    "Left_Hip_Internal_External_Rotation[rad]",
    "Left_Knee_Flexion_Extension[rad]",
    "Left_Ankle_Plantarflexion_Dorsiflexion[rad]",
    "Left_Ankle_Inversion_Eversion[rad]",
    "Lumbar_Flexion_Extension[rad]", "Lumbar_Lateral_Bending[rad]",
    "Left_Clavicle_Elevation_Depression[rad]",
    "Left_Shoulder_Flexion_Extension[rad]",
    "Left_Shoulder_Abduction_Adduction[rad]",
    "Left_Shoulder_Internal_External_Rotation[rad]",
    "Left_Elbow_Flexion_Extension[rad]", "Left_Elbow_Pronation_Supination[rad]",
    "Cervical_Flexion_Extension[rad]", "Cervical_Lateral_Bending[rad]",
    "Cervical_Internal_External_Rotation[rad]",
    "Right_Clavicle_Elevation_Depression[rad]",
    "Right_Shoulder_Flexion_Extension[rad]",
    "Right_Shoulder_Abduction_Adduction[rad]",
    "Right_Shoulder_Internal_External_Rotation[rad]",
    "Right_Elbow_Flexion_Extension[rad]",
    "Right_Elbow_Pronation_Supination[rad]",
    "Right_Hip_Flexion_Extension[rad]",
    "Right_Hip_Abduction_Adduction[rad]",
    "Right_Hip_Internal_External_Rotation[rad]",
    "Right_Knee_Flexion_Extension[rad]",
    "Right_Ankle_Plantarflexion_Dorsiflexion[rad]",
    "Right_Ankle_Inversion_Eversion[rad]",
]


# ── Acados MPC IK Solver ──────────────────────────────────────────────────

class AcadosMPCIKSolver:
    """Sliding-window (N-step) MPC-style IK solver using acados.

    State: x = [q; dq] (nq + nv)
    Control: u = ddq (nv)
    Dynamics:
    q_next = integrate(q, dq * dt)
    dq_next = dq + u * dt

    Stage cost (k=0..N-1): w_markers ||markers(q_k) - p[k]||²
    + w_dq ||dq_k||²
    + w_u ||u_k||²
    Terminal cost: w_dq ||dq_N||²

    Causal formulation: window = past N marker frames [p[i-N+1], ..., p[i]].
    Warm-started by shifting the previous optimal trajectory by 1.
    Output: q_{N-1} (state at last stage = current frame i).

    Uses full SQP (not SQP_RTI) for robustness. SQP_RTI's single iteration
    is insufficient for cold-start (neutral pose → first frame) and causes
    the QP to hit minimum step size, leading to divergence.
    """

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
        self._n_markers = len(keys_to_track)
        self._nmc = 3 * self._n_markers

        self._w_markers = w_markers
        self._w_dq = w_dq
        self._w_u = w_u

        # CasADi symbolic model via pinocchio.casadi
        self._cmodel = cpin.Model(self._pin_model)
        self._cdata = self._cmodel.createData()

        # Build acados OCP solver
        self._ocp_solver = self._create_ocp_solver()

    def _build_marker_fk_expr(self, cq):
        """CasADi expression for marker positions given symbolic q."""
        cpin.framesForwardKinematics(self._cmodel, self._cdata, cq)
        markers_expr = []
        for key in self._keys_to_track:
            frame_id = self._cmodel.getFrameId(key)
            if frame_id < len(self._pin_model.frames.tolist()):
                markers_expr.append(self._cdata.oMf[frame_id].translation)
        return casadi.vertcat(*markers_expr)

    def _create_ocp_solver(self):
        nmc = self._nmc

        # ── CasADi symbolics ──
        cx = casadi.SX.sym("x", self._nx)
        cu = casadi.SX.sym("u", self._nu)
        cq = cx[: self._nq]
        cdq = cx[self._nq:]

        # Discrete dynamics (Euler)
        q_next = cpin.integrate(self._cmodel, cq, cdq * self._dt)
        dq_next = cdq + cu * self._dt
        x_next = casadi.vertcat(q_next, dq_next)

        # Marker FK on current q
        markers_expr = self._build_marker_fk_expr(cq)

        # ── Acados model ──
        model = AcadosModel()
        model.name = f"human_ik_mpc_N{self._N}"
        model.x = cx
        model.u = cu
        model.disc_dyn_expr = x_next

        # Stage cost: y = [markers(q), dq, u]
        model.cost_y_expr = casadi.vertcat(markers_expr, cdq, cu)
        # Terminal cost: y_e = [dq]
        model.cost_y_expr_e = cdq

        # Parameter placeholder (unused in expression, but needed for interface)
        p = casadi.SX.sym("p", nmc)
        model.p = p

        # ── Acados OCP ──
        ocp = AcadosOcp()
        ocp.model = model

        ocp.solver_options.N_horizon = self._N
        ocp.solver_options.tf = self._N * self._dt

        # ── Cost setup ──
        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.cost_type_e = "NONLINEAR_LS"

        # Stage cost dimensions: [markers (nmc); dq (nv); u (nu)]
        ny = nmc + self._nv + self._nu
        ocp.cost.yref = np.zeros(ny)
        W = np.zeros((ny, ny))
        W[:nmc, :nmc] = self._w_markers * np.eye(nmc)
        W[nmc:nmc + self._nv, nmc:nmc + self._nv] = self._w_dq * np.eye(self._nv)
        W[nmc + self._nv:, nmc + self._nv:] = self._w_u * np.eye(self._nu)
        ocp.cost.W = W

        # Terminal cost: just dq regul
        ny_e = self._nv
        ocp.cost.yref_e = np.zeros(ny_e)
        ocp.cost.W_e = self._w_dq * np.eye(ny_e)

        # Joint limit constraints on q (skip freeflyer 0..6)
        n_joint_constraints = self._nq - 7
        if n_joint_constraints > 0:
            q_constrained = cx[7:self._nq]
            model.con_h_expr = q_constrained
            lh = np.array(self._pin_model.lowerPositionLimit[7:self._nq])
            uh = np.array(self._pin_model.upperPositionLimit[7:self._nq])
            ocp.constraints.lh = lh
            ocp.constraints.uh = uh

        # Initial state bounds (will be updated at each solve — we use them
        # to softly bias x_0 toward warm-start; keep unbounded by default)
        ocp.constraints.x0 = np.zeros(self._nx)

        # Parameter placeholder
        ocp.parameter_values = np.zeros(nmc)

        # ── Solver options ──
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "DISCRETE"
        # Full SQP (not SQP_RTI) for robustness during cold-start and
        # when markers change rapidly. RTI's single iteration isn't enough.
        ocp.solver_options.nlp_solver_type = "SQP"
        ocp.solver_options.nlp_solver_max_iter = 50
        ocp.solver_options.qp_solver_iter_max = 100
        ocp.solver_options.tol = 1e-4
        ocp.solver_options.globalization = "MERIT_BACKTRACKING"

        os.environ["ACADOS_SOURCE_DIR"] = str(
            Path(__file__).resolve().parent.parent / "acados"
        )

        ocp_solver = AcadosOcpSolver(ocp)
        return ocp_solver

    def reset_warm_start(self, q_init: np.ndarray):
        """Initialize the internal warm-start state with a given q."""
        x0 = np.zeros(self._nx)
        x0[:self._nq] = q_init
        for k in range(self._N + 1):
            self._ocp_solver.set(k, "x", x0)
        for k in range(self._N):
            self._ocp_solver.set(k, "u", np.zeros(self._nu))
        self._x0_warm = x0

    def solve(self, marker_window: list) -> np.ndarray:
        """Solve MPC IK for the current frame (causal, past N frames).

        Args:
        marker_window: list of N dicts with past N frames, oldest first.
        marker_window[k] is the measurement at stage k,
        corresponding to frame (i - N + 1 + k).
        marker_window[-1] = current frame i.

        Returns:
        q_{N-1}: Optimized joint configuration at current frame (nq,).
        """
        assert len(marker_window) == self._N, \
            f"Expected {self._N} frames in window, got {len(marker_window)}"

        # Hard-fix x_0 to the current warm-start (which is the shifted previous
        # solve's x_1). This anchors the trajectory to the previously committed
        # estimate, preventing drift to spurious local minima.
        x0_warm = self._ocp_solver.get(0, "x")
        self._ocp_solver.constraints_set(0, "lbx", x0_warm)
        self._ocp_solver.constraints_set(0, "ubx", x0_warm)

        # Set yref for each stage: [p_meas_k, 0 (dq), 0 (u)]
        for k in range(self._N):
            p_meas = np.zeros(self._nmc)
            for i, key in enumerate(self._keys_to_track):
                p_meas[3 * i: 3 * i + 3] = np.array(marker_window[k][key]).flatten()

            yref_k = np.concatenate([p_meas, np.zeros(self._nv), np.zeros(self._nu)])
            self._ocp_solver.set(k, "yref", yref_k)
            self._ocp_solver.set(k, "p", p_meas)

        # Terminal yref (dq target = 0, no marker)
        self._ocp_solver.set(self._N, "yref", np.zeros(self._nv))

        # Solve
        self._ocp_solver.solve()

        # Output: x_{N-1} (last stage with marker cost = current frame i)
        x_sol = self._ocp_solver.get(self._N - 1, "x")
        q_sol = x_sol[:self._nq]

        # Receding horizon shift: new x_k ← old x_{k+1}, new u_k ← old u_{k+1}
        # Window slides +1 frame in time, so the optimal trajectory shifts by -1 stage.
        for k in range(self._N):
            x_k_next = self._ocp_solver.get(k + 1, "x")
            self._ocp_solver.set(k, "x", x_k_next)
            if k < self._N - 1:
                u_k_next = self._ocp_solver.get(k + 1, "u")
                self._ocp_solver.set(k, "u", u_k_next)
        # Last stage keeps its previous value as a reasonable guess

        return q_sol


# ── Model loading (same as run_ik_acados.py) ──────────────────────────────

def load_and_prepare_model(
    model_dir: Path,
    urdf_file: str,
    comfi_root: Path,
    subject_id: str,
    start_sample_dict: dict,
    with_hand: bool,
    gender: str,
    subject_height: float,
) -> tuple:
    urdf_path = model_dir / urdf_file
    print(f"[INFO] Loading URDF from {urdf_path}")
    human = Robot(str(urdf_path), str(model_dir), isFext=True)
    human_model = human.model
    human_collision_model = human.collision_model
    human_visual_model = human.visual_model

    print(f"[INFO] Scaling human model (with_hand={with_hand})")
    human_model = scale_human_model(
        human_model, start_sample_dict,
        with_hand=with_hand, gender=gender, subject_height=subject_height,
    )
    print(f"[INFO] Model DOF (before locking): {human_model.nq}")

    print("[INFO] Registering markers to model")
    human_model = mks_registration(human_model, start_sample_dict, with_hand=False)

    print(f"[INFO] Locking {len(JOINTS_TO_LOCK)} joints")
    joint_ids_to_lock = []
    for jn in JOINTS_TO_LOCK:
        if human_model.existJointName(jn):
            joint_ids_to_lock.append(human_model.getJointId(jn))

    q0 = pin.neutral(human_model)
    human_model, human_visual_model = pin.buildReducedModel(
        human_model, human_visual_model, joint_ids_to_lock, q0,
    )
    print(f"[INFO] Model DOF (after locking): {human_model.nq}")

    return human_model, human_collision_model, human_visual_model


# ── Main ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Sliding-window MPC IK solver using acados"
    )
    p.add_argument("--id", dest="subject_id", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--comfi-root", default=Path(os.environ.get("COMFI_ROOT", "COMFI")), type=Path)
    p.add_argument("--output-root", default=Path("output").resolve() / "res_hpe", type=Path)
    p.add_argument("--model-dir", default=Path("model"), type=Path)
    p.add_argument("--urdf-file", default="urdf/human.urdf")
    p.add_argument("--augmented-file", default="augmented_markers.csv")
    p.add_argument("--keypoints-file", default="3d_keypoints_4cams.csv")
    p.add_argument("--start-sample", type=int, default=0)
    p.add_argument("--with-hand", action="store_true")
    p.add_argument("--dt", type=float, default=0.025, help="Time step (1/fps). Default: 0.025")
    p.add_argument("--N", type=int, default=10, help="MPC horizon length. Default: 10")
    p.add_argument("--w-markers", type=float, default=1.0)
    p.add_argument("--w-dq", type=float, default=1e-3)
    p.add_argument("--w-u", type=float, default=1e-4)
    p.add_argument("--display", action="store_true")
    p.add_argument("--save-video", action="store_true")
    p.add_argument("--meshcat-url", default="tcp://127.0.0.1:6000")
    p.add_argument("--output-angles-file", default="joint_angles_acados_mpc.csv")
    p.add_argument("--output-markers-file", default="markers_model_trajectories_acados_mpc.csv")
    return p.parse_args()


def main():
    args = parse_args()

    if args.subject_id not in SUBJECT_IDS:
        raise ValueError(f"Unknown subject ID: {args.subject_id}")
    if args.task not in TASKS:
        raise ValueError(f"Unknown task: {args.task}")

    comfi_root = Path(args.comfi_root).resolve()
    output_root = Path(args.output_root).resolve()
    model_dir = Path(args.model_dir).resolve()

    metadata_yaml = comfi_root / "metadata" / f"{args.subject_id}.yaml"
    path_to_csv = output_root / args.subject_id / args.task / args.augmented_file
    path_to_kpt = output_root / args.subject_id / args.task / args.keypoints_file
    output_markers_csv = output_root / args.subject_id / args.task / args.output_markers_file
    output_angles_csv = output_root / args.subject_id / args.task / args.output_angles_file

    for path, desc in [
        (metadata_yaml, "metadata"), (path_to_csv, "augmented markers"),
        (path_to_kpt, "3D keypoints"), (model_dir / args.urdf_file, "URDF"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Missing {desc} file: {path}")

    _, subject_height, subject_mass, gender = read_subject_yaml(str(metadata_yaml))
    print(f"[INFO] Subject {args.subject_id}: height={subject_height}m, mass={subject_mass}kg, gender={gender}")

    print("[INFO] Loading marker data...")
    data_markers_lstm = pd.read_csv(path_to_csv)
    keypoints = pd.read_csv(path_to_kpt) / 1000 # mm -> m
    columns_to_add = [
        col for col in keypoints.columns if any(key + "_" in col for key in KEYS_TO_ADD)
    ]
    mks_data = pd.concat(
        [data_markers_lstm, keypoints[columns_to_add].reset_index(drop=True)], axis=1,
    )
    result_markers, start_sample_dict = read_mks_data(mks_data, start_sample=args.start_sample)
    print(f"[INFO] Loaded {len(result_markers)} frames")

    human_model, human_collision_model, human_visual_model = load_and_prepare_model(
        model_dir, args.urdf_file, comfi_root, args.subject_id,
        start_sample_dict, args.with_hand, gender, subject_height,
    )
    human_data = pin.Data(human_model)

    # ── Visualization setup ──
    viz = None
    viewer = None
    if args.display:
        print(f"[INFO] Initializing Meshcat visualizer at {args.meshcat_url}")
        viz = MeshcatVisualizer(human_model, human_collision_model, human_visual_model)
        viewer = meshcat.Visualizer(zmq_url=args.meshcat_url)
        viz.initViewer(viewer=viewer, open=True)
        viz.viewer.delete()
        viz.loadViewerModel("human_model")

        viewer["/Background"].set_property("top_color", [1, 1, 1])
        viewer["/Background"].set_property("bottom_color", [0.65, 0.65, 0.65])

        for marker_name in KEYS_TO_TRACK:
            viewer[f"gt_markers/{marker_name}"].set_object(
                meshcat.geometry.Sphere(0.02),
                meshcat.geometry.MeshLambertMaterial(color=0xFF0000, opacity=0.4),
            )
            viewer[f"model_markers/{marker_name}"].set_object(
                meshcat.geometry.Sphere(0.015),
                meshcat.geometry.MeshLambertMaterial(color=0x000000, opacity=0.4),
            )

        q0 = pin.neutral(human_model)
        viz.display(q0)
        first_mks = result_markers[args.start_sample]
        for marker_name in KEYS_TO_TRACK:
            if marker_name in first_mks:
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
        keys_to_track=KEYS_TO_TRACK,
        N=args.N,
        dt=args.dt,
        w_markers=args.w_markers,
        w_dq=args.w_dq,
        w_u=args.w_u,
    )

    # Warm-start: solve first frame with IPOPT to get a reasonable initial pose.
    # Without this, MPC starts from neutral pose (lying flat) and the hard-fixed
    # x_0 would trap the solver in a bad region.
    print("[INFO] Warm-starting first-frame IK with IPOPT...")
    from comfi_examples.ik_utils import RT_IK
    omega = {key: 1.0 for key in KEYS_TO_TRACK}
    q0 = pin.neutral(human_model)
    rt_ik = RT_IK(
        human_model, result_markers[args.start_sample], q0,
        KEYS_TO_TRACK, args.dt, omega,
    )
    q_init = rt_ik.solve_ik_sample_casadi()
    print(f"[INFO] First-frame IK done, initializing MPC warm-start")
    ik_solver.reset_warm_start(q_init)

    # ── Main MPC loop (causal, past N frames) ──
    # For each output frame i, use window [i-N+1, ..., i] (past N frames).
    # For i < N-1, pad the window with the earliest available frame.
    n_frames = len(result_markers)
    print(f"[INFO] Solving MPC IK for frames [{args.start_sample}, {n_frames})...")

    q_list = []
    M_model_list = []
    rmse_per_marker = {}
    solve_times = []
    video_frames = [] if (args.display and args.save_video) else None

    for ii in range(args.start_sample, n_frames):
        # Build window of N past marker frames (oldest first).
        # If i < N-1, pad with the earliest available frame.
        marker_window = [
            result_markers[max(args.start_sample, ii - args.N + 1 + k)]
            for k in range(args.N)
        ]
        current_mks = marker_window[-1] # latest frame = current frame i

        # Solve
        t0 = time.perf_counter()
        q = ik_solver.solve(marker_window)
        solve_times.append(time.perf_counter() - t0)

        # FK for evaluation
        pin.forwardKinematics(human_model, human_data, q)
        pin.updateFramePlacements(human_model, human_data)

        M_model_frame = {}
        for marker in current_mks.keys():
            if marker in MKS_TO_SKIP:
                continue
            pos_gt = np.array(current_mks[marker]).flatten()
            M_model = human_data.oMf[human_model.getFrameId(marker)]
            pos_model = np.array(M_model.translation).flatten()

            M_model_frame[f"{marker}_x"] = M_model.translation[0]
            M_model_frame[f"{marker}_y"] = M_model.translation[1]
            M_model_frame[f"{marker}_z"] = M_model.translation[2]

            if viewer and marker in KEYS_TO_TRACK:
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

    # ── Save video ──
    if video_frames is not None and len(video_frames) > 0:
        video_path = output_root / args.subject_id / args.task / "ik_acados_mpc.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        fps = int(1.0 / args.dt)
        imageio.mimsave(str(video_path), video_frames, fps=fps)
        print(f"[INFO] Saved visualization video to {video_path} ({len(video_frames)} frames, {fps} fps)")

    # ── Solve time statistics ──
    solve_times_arr = np.array(solve_times)
    print("\n" + "=" * 60)
    print(f"MPC IK SOLVE TIME (N={args.N}, per frame):")
    print("=" * 60)
    print(f" frames : {len(solve_times_arr)}")
    print(f" mean : {solve_times_arr.mean() * 1e3:.2f} ms")
    print(f" median : {np.median(solve_times_arr) * 1e3:.2f} ms")
    print(f" std : {solve_times_arr.std() * 1e3:.2f} ms")
    print(f" min / max : {solve_times_arr.min() * 1e3:.2f} / {solve_times_arr.max() * 1e3:.2f} ms")
    print(f" total : {solve_times_arr.sum():.2f} s")
    print("=" * 60 + "\n")

    output_times_csv = output_angles_csv.parent / "solve_times_acados_mpc.csv"
    pd.DataFrame({
        "frame": np.arange(args.start_sample, args.start_sample + len(solve_times_arr)),
        "solve_time_s": solve_times_arr,
    }).to_csv(output_times_csv, index=False)
    print(f"[INFO] Saved per-frame solve times to {output_times_csv}")

    # Save results
    print(f"[INFO] Saving marker trajectories to {output_markers_csv}")
    output_markers_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(M_model_list).to_csv(output_markers_csv, index=False)

    print(f"[INFO] Saving joint angles to {output_angles_csv}")
    if len(JOINT_ANGLES_NAMES) != len(q_list[0]):
        raise ValueError(
            f"JOINT_ANGLES_NAMES has {len(JOINT_ANGLES_NAMES)} entries but q has {len(q_list[0])} DOFs."
        )
    pd.DataFrame(q_list, columns=JOINT_ANGLES_NAMES).to_csv(output_angles_csv, index=False)

    # Print RMSE
    print("\n" + "=" * 60)
    print("TRACKING ERROR - Per-marker RMSE (meters):")
    print("=" * 60)
    rmse_global = 0
    nb_mks = 0
    for marker, sq_errors in sorted(rmse_per_marker.items()):
        nb_mks += 1
        rmse = np.sqrt(np.mean(sq_errors))
        print(f" {marker:20s}: {rmse:.4f} m")
    rmse_global += rmse
    rmse_global /= nb_mks
    print("=" * 60)
    print(f"GLOBAL RMSE (all markers): {rmse_global:.4f} m")
    print("=" * 60 + "\n")

    print(f"[SUCCESS] {args.subject_id}/{args.task} acados MPC IK processing complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())