#!/usr/bin/env python3

"""Sliding-window MPC-style inverse kinematics solver using acados.

The IK is formulated as a multi-stage OCP over a horizon of N frames:
State: x = [q; dq] (nq + nv)
Control: u = ddq (nv)
Dynamics: q_next = integrate(q, dq * dt)
dq_next = dq + ddq * dt
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

#****************************************************************************
SHARPA_FRAMES = [
    # --- THUMB ---
    "right_thumb_fingertip", "right_thumb_DP", "right_thumb_PP", "right_thumb_MC",
    # --- INDEX ---
    "right_index_fingertip", "right_index_DP", "right_index_MP", "right_index_PP",
    # --- MIDDLE ---
    "right_middle_fingertip", "right_middle_DP", "right_middle_MP", "right_middle_PP",
    # --- RING ---
    "right_ring_fingertip", "right_ring_DP", "right_ring_MP", "right_ring_PP",
    # --- PINKY ---
    "right_pinky_fingertip", "right_pinky_DP", "right_pinky_MP", "right_pinky_PP",
    # --- WRIST ---
    "right_hand_wrist" 
]

MANO_FRAMES = [
    # --- THUMB (tip=0, dip=1, pip=2, mcp=3) ---
    "thumb3", "thumb2", "thumb1z", "thumb1y",
    # --- INDEX (tip=4, dip=5, pip=6, mcp=7) ---
    "index3", "index2", "index1x", "index1y",
    # --- MIDDLE (tip=8, dip=9, pip=10, mcp=11) ---
    "middle3", "middle2", "middle1x", "middle1y",
    # --- RING (tip=12, dip=13, pip=14, mcp=15) ---
    "ring3", "ring2", "ring1x", "ring1y",
    # --- PINKY (tip=16, dip=17, pip=18, mcp=19) ---
    "pinky3", "pinky2", "pinky1x", "pinky1y",
    # --- WRIST (wrist=20) ---
    "right_hand_wrist"
]
#****************************************************************************


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
        self._n_markers = len(keys_to_track)
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
        for key in self._keys_to_track:
            frame_id = self._cmodel.getFrameId(key)
            if frame_id < len(self._pin_model.frames.tolist()):
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

        n_joint_constraints = self._nq - 4
        if n_joint_constraints > 0:
            q_constrained = cx[4:self._nq]
            model.con_h_expr = q_constrained
            lh = np.array(self._pin_model.lowerPositionLimit[4:self._nq])
            uh = np.array(self._pin_model.upperPositionLimit[4:self._nq])
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
        ocp_solver = AcadosOcpSolver(ocp)
        return ocp_solver

    def reset_warm_start(self, q_init: np.ndarray):
        x0 = np.zeros(self._nx)
        x0[:self._nq] = q_init
        for k in range(self._N + 1):
            self._ocp_solver.set(k, "x", x0)
        for k in range(self._N):
            self._ocp_solver.set(k, "u", np.zeros(self._nu))
        self._x0_warm = x0

    def solve(self, marker_window: list) -> np.ndarray:
        assert len(marker_window) == self._N, f"Expected {self._N} frames, got {len(marker_window)}"
        x0_warm = self._ocp_solver.get(0, "x")
        self._ocp_solver.constraints_set(0, "lbx", x0_warm)
        self._ocp_solver.constraints_set(0, "ubx", x0_warm)

        for k in range(self._N):
            p_meas = np.zeros(self._nmc)
            for i, key in enumerate(self._keys_to_track):
                p_meas[3 * i: 3 * i + 3] = np.array(marker_window[k][key]).flatten()

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


# ── Model loading ─────────────────────────────────────────────────────────

def load_sharpa_model(urdf_path: str, model_dir: str):
    print(f"[INFO] Loading URDF from {urdf_path}")
    root_joint = pin.JointModelSpherical()
    
    mano_dir = os.path.join(model_dir, "mano-urdf")
    mano_mesh_dir = os.path.join(model_dir, "mano-urdf", "urdf") 
    wave_01_dir = os.path.join(model_dir, "sharpa_hand", "wave_01")
    
    model, collision_model, visual_model = pin.buildModelsFromUrdf(
        urdf_path, 
        root_joint=root_joint, 
        package_dirs=[model_dir, wave_01_dir, mano_dir, mano_mesh_dir] 
    )
    print(f"[INFO] Model DOF: {model.nq} (Spherical Base)")
    return model, collision_model, visual_model


# ── Main ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Sliding-window MPC IK solver using acados")
    p.add_argument("--id", dest="subject_id", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--comfi-root", default=Path(os.environ.get("COMFI_ROOT", "COMFI")), type=Path)
    p.add_argument("--output-root", default=Path("output").resolve() / "res_hpe", type=Path)
    p.add_argument("--model-dir", default=Path("model"), type=Path)
    p.add_argument("--urdf-file", default="urdf/human.urdf")
    p.add_argument("--augmented-file", default="augmented_markers.csv")
    p.add_argument("--keypoints-file", default="dexsuite_joints1.npy")
    p.add_argument("--start-sample", type=int, default=0)
    p.add_argument("--with-hand", action="store_true")
    p.add_argument("--dt", type=float, default=0.025)
    p.add_argument("--N", type=int, default=10)
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

    output_root = Path(args.output_root).resolve()
    model_dir = Path(args.model_dir).resolve()

    path_to_kpt = output_root / args.subject_id / args.task / args.keypoints_file
    output_markers_csv = output_root / args.subject_id / args.task / args.output_markers_file
    output_angles_csv = output_root / args.subject_id / args.task / args.output_angles_file
    sharpa_urdf = str(model_dir / args.urdf_file)

    if not path_to_kpt.exists():
        raise FileNotFoundError(f"Missing target Numpy data: {path_to_kpt}")
    if not Path(shar_urdf := sharpa_urdf).exists():
        raise FileNotFoundError(f"Missing URDF: {sharpa_urdf}")

    #********************************************************************************
    # DYNAMIC CONFIGURATION: Switch tracking dictionary based on loaded URDF file name
    #********************************************************************************
    is_mano = "mano" in args.urdf_file.lower()
    if is_mano:
        active_tracking_frames = MANO_FRAMES
        print("[INFO] Target model identified as MANO hand")
    else:
        active_tracking_frames = SHARPA_FRAMES
        print("[INFO] Target model identified as SHARPA hand")

    # 1. Load the Model
    human_model, human_collision_model, human_visual_model = load_sharpa_model(sharpa_urdf, str(model_dir))    
    human_data = pin.Data(human_model)

    # 2. Load SAM3D Numpy Data
    print(f"[INFO] Loading target keypoints from {path_to_kpt}...")
    sam3d_data = np.load(str(path_to_kpt)) 
    
    right_hand_points = sam3d_data[:, 1, :, :]
    right_hand_points[:,:,2] = -right_hand_points[:,:,2]  
    n_frames = right_hand_points.shape[0]
    
    global_scale = 1
    right_hand_points = right_hand_points * global_scale

    # Center the entire hand around the origin [0,0,0]
    wrist_positions = right_hand_points[:, 20:21, :].copy()
    right_hand_points = right_hand_points - wrist_positions
    
    # ROTATE THE DATA TO MATCH THE ROBOT
    import math
    roll = math.pi/2
    pitch = -math.pi 
    yaw = 0
    R_align = pin.rpy.rpyToMatrix(roll, pitch, yaw)
    
    for i in range(n_frames):
        right_hand_points[i] = right_hand_points[i] @ R_align.T

    # Convert mapping list using active configuration array
    result_markers = []
    for frame_idx in range(n_frames):
        frame_dict = {}
        for kp_idx, frame_name in enumerate(active_tracking_frames):
            frame_dict[frame_name] = right_hand_points[frame_idx, kp_idx, :]
        result_markers.append(frame_dict)

    print(f"[INFO] Prepared {n_frames} frames for IK tracking")
    #********************************************************************************

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

        for marker_name in active_tracking_frames:
            viewer[f"gt_markers/{marker_name}"].set_object(
                meshcat.geometry.Sphere(0.01),
                meshcat.geometry.MeshLambertMaterial(color=0xFF0000, opacity=0.6),
            )
            viewer[f"model_markers/{marker_name}"].set_object(
                meshcat.geometry.Sphere(0.008),
                meshcat.geometry.MeshLambertMaterial(color=0x000000, opacity=0.6),
            )

        # ──────────────────────────────────────────────────────────────────────
        # INITIALIZE WARM-START Trajectory Profile (q_init)
        # ──────────────────────────────────────────────────────────────────────
        q_init = pin.neutral(human_model)
        
        if is_mano:
            import math
            # Calculate a base quaternion rotation that aligns the palm face 
            # with the initial posture of the tracking camera data
            # Adjust these RPY angles to match your specific tracking space:
            base_rpy = np.array([math.pi / 2, math.pi/2, math.pi])
            base_quat = pin.Quaternion(pin.rpy.rpyToMatrix(base_rpy))
            
            # Spherical root joints map their orientation to the first 4 elements [x, y, z, w]
            q_init[0] = base_quat.x
            q_init[1] = base_quat.y
            q_init[2] = base_quat.z
            q_init[3] = base_quat.w
            
            if len(q_init) > 4:
                q_init[4:] = 0.1
        else:
            # Sharpa default open configuration
            if len(q_init) > 4:
                # Tweak specific indices here if the thumb base conflicts with initial targets
                pass

        viz.display(q_init)
        first_mks = result_markers[max(0, args.start_sample)]
        for marker_name in active_tracking_frames:
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
        keys_to_track=active_tracking_frames, 
        N=args.N,
        dt=args.dt,
        w_markers=args.w_markers,
        w_dq=args.w_dq,
        w_u=args.w_u,
    )

    print("[INFO] Warm-starting Spherical Base Trajectory Horizon...")
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
            pos_gt = np.array(current_mks[marker]).flatten()
            M_model = human_data.oMf[human_model.getFrameId(marker)]
            pos_model = np.array(M_model.translation).flatten()

            M_model_frame[f"{marker}_x"] = M_model.translation[0]
            M_model_frame[f"{marker}_y"] = M_model.translation[1]
            M_model_frame[f"{marker}_z"] = M_model.translation[2]

            if viewer and marker in active_tracking_frames:
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

    q_arr = np.array(q_list)
    zeros_trans = np.zeros((q_arr.shape[0], 3))
    q_ff_arr = np.hstack((zeros_trans, q_arr))

    pd.DataFrame(q_ff_arr).to_csv(output_angles_csv, index=False)

    print(f"[SUCCESS] {args.subject_id}/{args.task} acados MPC IK processing complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())