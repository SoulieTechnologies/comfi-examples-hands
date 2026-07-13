#!/usr/bin/env python3
"""SAM3D full-body + MANO hand inverse kinematics via ACADOS MPC.

Takes 70-point Goliath keypoints from SAM3D and solves full-body IK
(including MANO finger articulation) on human_with_mano.urdf.

State:   x = [q; dq]      (nq + nv)
Control: u = ddq           (nv)
Dynamics: q+ = integrate(q, dq*dt),  dq+ = dq + u*dt
Cost:     sum_k  w*||FK(q_k) - p_k||^2 + w_dq*||dq_k||^2 + w_u*||u_k||^2

Goliath layout (70 keypoints):
    0-20   core body (21 kp, no wrists)
    21-41  right hand (21 kp, wrist = 41)
    42-62  left hand  (21 kp, wrist = 62)
    63-69  extra body (olecranons, cubital fossae, acromions, neck)

Requires: pinocchio, pinocchio.casadi, casadi, acados_template
"""

import os
import sys
import argparse
import time
from pathlib import Path

import casadi
import numpy as np
import pandas as pd
import pinocchio as pin
import pinocchio.casadi as cpin
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from comfi_examples.urdf_utils import scale_human_model
from comfi_examples.augmenter_utils import augmentTRC, loadModel


# ═══════════════════════════════════════════════════════════════════════════
# GOLIATH KEYPOINT NAMES  (70 points, raw index order from Meta Sapiens)
# ═══════════════════════════════════════════════════════════════════════════

GOLIATH_NAMES = [
    # Core body 0-20
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_big_toe", "left_small_toe", "left_heel",
    "right_big_toe", "right_small_toe", "right_heel",
    # Right hand 21-41
    "right_thumb4", "right_thumb3", "right_thumb2",
    "right_thumb_third_joint",
    "right_forefinger4", "right_forefinger3", "right_forefinger2",
    "right_forefinger_third_joint",
    "right_middle_finger4", "right_middle_finger3", "right_middle_finger2",
    "right_middle_finger_third_joint",
    "right_ring_finger4", "right_ring_finger3", "right_ring_finger2",
    "right_ring_finger_third_joint",
    "right_pinky_finger4", "right_pinky_finger3", "right_pinky_finger2",
    "right_pinky_finger_third_joint",
    "right_wrist",
    # Left hand 42-62
    "left_thumb4", "left_thumb3", "left_thumb2",
    "left_thumb_third_joint",
    "left_forefinger4", "left_forefinger3", "left_forefinger2",
    "left_forefinger_third_joint",
    "left_middle_finger4", "left_middle_finger3", "left_middle_finger2",
    "left_middle_finger_third_joint",
    "left_ring_finger4", "left_ring_finger3", "left_ring_finger2",
    "left_ring_finger_third_joint",
    "left_pinky_finger4", "left_pinky_finger3", "left_pinky_finger2",
    "left_pinky_finger_third_joint",
    "left_wrist",
    # Extra body 63-69
    "left_olecranon", "right_olecranon",
    "left_cubital_fossa", "right_cubital_fossa",
    "left_acromion", "right_acromion", "neck",
]


# ═══════════════════════════════════════════════════════════════════════════
# FRAME REGISTRATION  —  Goliath keypoint name -> URDF parent link
# ═══════════════════════════════════════════════════════════════════════════

# Body keypoints (core 0-20 + neck 69) -> body links at zero local offset
GOLIATH_BODY_TO_LINK = {
    "nose":             "middle_head",
    "left_eye":         "middle_head",
    "right_eye":        "middle_head",
    "left_ear":         "middle_head",
    "right_ear":        "middle_head",
    "left_shoulder":    "left_upperarm",
    "right_shoulder":   "right_upperarm",
    "left_elbow":       "left_lowerarm",
    "right_elbow":      "right_lowerarm",
    "left_hip":         "left_upperleg",
    "right_hip":        "right_upperleg",
    "left_knee":        "left_lowerleg",
    "right_knee":       "right_lowerleg",
    "left_ankle":       "left_foot",
    "right_ankle":      "right_foot",
    "left_big_toe":     "left_foot",
    "left_small_toe":   "left_foot",
    "left_heel":        "left_foot",
    "right_big_toe":    "right_foot",
    "right_small_toe":  "right_foot",
    "right_heel":       "right_foot",
    "neck":             "middle_thorax",
}

# Hand keypoints (21 right + 21 left) -> MANO links
HAND_KP_TO_MANO_LINK = {
    # Right hand
    "right_thumb4":                     "mano_right_thumb3",
    "right_thumb3":                     "mano_right_thumb2",
    "right_thumb2":                     "mano_right_thumb1z",
    "right_thumb_third_joint":          "mano_right_thumb1y",
    "right_forefinger4":                "mano_right_index3",
    "right_forefinger3":                "mano_right_index2",
    "right_forefinger2":                "mano_right_index1x",
    "right_forefinger_third_joint":     "mano_right_index1y",
    "right_middle_finger4":             "mano_right_middle3",
    "right_middle_finger3":             "mano_right_middle2",
    "right_middle_finger2":             "mano_right_middle1x",
    "right_middle_finger_third_joint":  "mano_right_middle1y",
    "right_ring_finger4":               "mano_right_ring3",
    "right_ring_finger3":               "mano_right_ring2",
    "right_ring_finger2":               "mano_right_ring1x",
    "right_ring_finger_third_joint":    "mano_right_ring1y",
    "right_pinky_finger4":              "mano_right_pinky3",
    "right_pinky_finger3":              "mano_right_pinky2",
    "right_pinky_finger2":              "mano_right_pinky1x",
    "right_pinky_finger_third_joint":   "mano_right_pinky1y",
    "right_wrist":                      "mano_right_right_hand_wrist",
    # Left hand
    "left_thumb4":                      "mano_left_thumb3",
    "left_thumb3":                      "mano_left_thumb2",
    "left_thumb2":                      "mano_left_thumb1z",
    "left_thumb_third_joint":           "mano_left_thumb1y",
    "left_forefinger4":                 "mano_left_index3",
    "left_forefinger3":                 "mano_left_index2",
    "left_forefinger2":                 "mano_left_index1x",
    "left_forefinger_third_joint":      "mano_left_index1y",
    "left_middle_finger4":              "mano_left_middle3",
    "left_middle_finger3":              "mano_left_middle2",
    "left_middle_finger2":              "mano_left_middle1x",
    "left_middle_finger_third_joint":   "mano_left_middle1y",
    "left_ring_finger4":                "mano_left_ring3",
    "left_ring_finger3":                "mano_left_ring2",
    "left_ring_finger2":                "mano_left_ring1x",
    "left_ring_finger_third_joint":     "mano_left_ring1y",
    "left_pinky_finger4":               "mano_left_pinky3",
    "left_pinky_finger3":               "mano_left_pinky2",
    "left_pinky_finger2":               "mano_left_pinky1x",
    "left_pinky_finger_third_joint":    "mano_left_pinky1y",
    "left_wrist":                       "mano_left_left_hand_wrist",
}

# Fingertip distal-phalanx tip offsets (local frame, meters)
_TIP_OFFSETS = {
    "right_thumb4":          np.array([0., 0., -0.018]),
    "right_forefinger4":     np.array([0., 0., -0.015]),
    "right_middle_finger4":  np.array([0., 0., -0.016]),
    "right_ring_finger4":    np.array([0., 0., -0.016]),
    "right_pinky_finger4":   np.array([0., 0., -0.013]),
    "left_thumb4":           np.array([0., 0.,  0.018]),
    "left_forefinger4":      np.array([0., 0.,  0.015]),
    "left_middle_finger4":   np.array([0., 0.,  0.016]),
    "left_ring_finger4":     np.array([0., 0.,  0.016]),
    "left_pinky_finger4":    np.array([0., 0.,  0.013]),
}


# ═══════════════════════════════════════════════════════════════════════════
# SOLVER KEYS  —  which keypoints the MPC actually tracks
# ═══════════════════════════════════════════════════════════════════════════

BODY_SOLVER_KEYS = [
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_big_toe", "right_big_toe",
    "left_heel", "right_heel",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "nose", "neck",
]

HAND_SOLVER_KEYS = GOLIATH_NAMES[21:63]  # all 42 hand keypoints

SOLVER_KEYS = BODY_SOLVER_KEYS + HAND_SOLVER_KEYS  # 58 total

JOINTS_TO_LOCK = [
    "middle_thoracic_X", "middle_thoracic_Y", "middle_thoracic_Z",
]


# ═══════════════════════════════════════════════════════════════════════════
# FRAME REGISTRATION
# ═══════════════════════════════════════════════════════════════════════════

def register_tracking_frames(model):
    """Add OP_FRAME for each Goliath keypoint on its URDF parent link.

    Body keypoints are registered at zero local offset (joint-center
    approximation).  Fingertip keypoints get a small distal-phalanx
    tip offset.  All other hand keypoints sit at the MANO link origin.

    Returns (model, list_of_registered_names).
    """
    inertia = pin.Inertia.Zero()
    all_kp_to_link = {}
    all_kp_to_link.update(GOLIATH_BODY_TO_LINK)
    all_kp_to_link.update(HAND_KP_TO_MANO_LINK)

    registered = []
    for kp_name, parent_link in all_kp_to_link.items():
        if not model.existFrame(parent_link):
            print(f"[WARN] Link '{parent_link}' not in URDF, "
                  f"skipping '{kp_name}'")
            continue

        parent_fid = model.getFrameId(parent_link)
        parent_jid = model.frames[parent_fid].parentJoint
        offset = _TIP_OFFSETS.get(kp_name, np.zeros(3))

        frame = pin.Frame(
            kp_name,
            parent_jid,
            parent_fid,
            pin.SE3(np.eye(3), offset),
            pin.FrameType.OP_FRAME,
            inertia,
        )
        model.addFrame(frame, False)
        registered.append(kp_name)

    return model, registered


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_sam3d_data(path, scale=1.0, cv_to_ros=True, tilt_deg=0.0):
    """Load SAM3D Goliath keypoints -> (N, 70, 3) in metres."""
    path = Path(path)
    if path.suffix == ".npy":
        data = np.load(str(path)).astype(np.float64)
        if data.ndim == 2:
            data = data.reshape(-1, 70, 3)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
        n = len(df)
        data = np.zeros((n, 70, 3))
        for i, name in enumerate(GOLIATH_NAMES):
            for j, ax in enumerate(["x", "y", "z"]):
                col = f"{name}_{ax}"
                if col in df.columns:
                    data[:, i, j] = df[col].values
                elif 3 * i + j < len(df.columns):
                    data[:, i, j] = df.iloc[:, 3 * i + j].values
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")

    if scale != 1.0:
        data *= scale

    # OpenCV (Right,Down,Forward) -> ROS/Pinocchio (Forward,Left,Up)
    if cv_to_ros:
        cv_x = data[:, :, 0].copy()
        cv_y = data[:, :, 1].copy()
        cv_z = data[:, :, 2].copy()
        data[:, :, 0] = cv_z    # X = Forward
        data[:, :, 1] = -cv_x   # Y = Left
        data[:, :, 2] = -cv_y   # Z = Up

    # Correct camera downward tilt: rotate around Y-axis (lateral) in ROS frame
    if tilt_deg != 0.0:
        a = np.radians(tilt_deg)
        Ry = np.array([[np.cos(a), 0, np.sin(a)],
                       [0,         1, 0         ],
                       [-np.sin(a), 0, np.cos(a)]])
        N, K, _ = data.shape
        data = (Ry @ data.reshape(-1, 3).T).T.reshape(N, K, 3)

    return data


# ═══════════════════════════════════════════════════════════════════════════
# ACADOS MPC IK SOLVER
# ═══════════════════════════════════════════════════════════════════════════

class AcadosMPCIKSolver:
    """Sliding-window MPC IK for body + MANO hands via acados.

    Supports separate marker weights for body vs hand keypoints.
    Only tracks keys that actually exist as frames in the model.
    """

    def __init__(
        self,
        pin_model,
        keys_to_track,
        N=10,
        dt=0.025,
        w_markers=1.0,
        w_hand=1.0,
        w_dq=1e-3,
        w_dq_hand=None,
        w_u=1e-4,
        hand_keys=None,
        w_fingertip_mult=2.0,
    ):
        self._pin_model = pin_model
        self._nq = pin_model.nq
        self._nv = pin_model.nv
        self._nx = self._nq + self._nv
        self._nu = self._nv
        self._N = N
        self._dt = dt

        self._valid_keys = [
            k for k in keys_to_track if pin_model.existFrame(k)
        ]
        if len(self._valid_keys) < len(keys_to_track):
            n_miss = len(keys_to_track) - len(self._valid_keys)
            print(f"[WARN] {n_miss} solver keys not found in model")

        self._n_markers = len(self._valid_keys)
        self._nmc = 3 * self._n_markers

        self._w_markers = w_markers
        self._w_hand = w_hand
        self._w_fingertip_mult = w_fingertip_mult
        self._w_dq = w_dq
        self._w_dq_hand = w_dq_hand if w_dq_hand is not None else w_dq
        self._w_u = w_u
        self._hand_keys = set(hand_keys or [])

        # Build per-DOF w_dq vector: w_dq_hand for MANO joints, w_dq for body
        w_dq_vec = np.full(self._nv, w_dq)
        for j in range(1, pin_model.njoints):
            if "mano" in pin_model.names[j]:
                idx_v = pin_model.joints[j].idx_v
                nv_j = pin_model.joints[j].nv
                w_dq_vec[idx_v:idx_v + nv_j] = self._w_dq_hand
        self._w_dq_vec = w_dq_vec

        self._cmodel = cpin.Model(self._pin_model)
        self._cdata = self._cmodel.createData()
        self._ocp_solver = self._create_ocp_solver()

    def _build_marker_fk_expr(self, cq):
        cpin.framesForwardKinematics(self._cmodel, self._cdata, cq)
        exprs = []
        for key in self._valid_keys:
            fid = self._cmodel.getFrameId(key)
            exprs.append(self._cdata.oMf[fid].translation)
        return casadi.vertcat(*exprs)

    def _create_ocp_solver(self):
        nmc = self._nmc
        cx = casadi.SX.sym("x", self._nx)
        cu = casadi.SX.sym("u", self._nu)
        cq, cdq = cx[:self._nq], cx[self._nq:]

        q_next = cpin.integrate(self._cmodel, cq, cdq * self._dt)
        dq_next = cdq + cu * self._dt
        x_next = casadi.vertcat(q_next, dq_next)
        markers_expr = self._build_marker_fk_expr(cq)

        model = AcadosModel()
        model.name = f"sam3d_mano_ik_N{self._N}"
        model.x = cx
        model.u = cu
        model.disc_dyn_expr = x_next
        model.cost_y_expr = casadi.vertcat(markers_expr, cdq, cu)
        model.cost_y_expr_e = cdq
        model.p = casadi.SX.sym("p", nmc)

        ocp = AcadosOcp()
        ocp.model = model
        ocp.solver_options.N_horizon = self._N
        ocp.solver_options.tf = self._N * self._dt

        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.cost_type_e = "NONLINEAR_LS"

        ny = nmc + self._nv + self._nu
        ocp.cost.yref = np.zeros(ny)

        W = np.zeros((ny, ny))
        for i, key in enumerate(self._valid_keys):
            w = self._w_hand if key in self._hand_keys else self._w_markers
            if key in self._hand_keys and key.endswith("4"):
                w *= self._w_fingertip_mult
            W[3 * i:3 * i + 3, 3 * i:3 * i + 3] = w * np.eye(3)
        W[nmc:nmc + self._nv, nmc:nmc + self._nv] = np.diag(self._w_dq_vec)
        W[nmc + self._nv:, nmc + self._nv:] = self._w_u * np.eye(self._nu)
        ocp.cost.W = W

        ocp.cost.yref_e = np.zeros(self._nv)
        ocp.cost.W_e = np.diag(self._w_dq_vec)

        njc = self._nq - 7
        if njc > 0:
            model.con_h_expr = cx[7:self._nq]
            ocp.constraints.lh = np.array(
                self._pin_model.lowerPositionLimit[7:self._nq])
            ocp.constraints.uh = np.array(
                self._pin_model.upperPositionLimit[7:self._nq])

        ocp.constraints.x0 = np.zeros(self._nx)
        ocp.parameter_values = np.zeros(nmc)

        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "DISCRETE"
        ocp.solver_options.nlp_solver_type = "SQP"
        ocp.solver_options.nlp_solver_max_iter = 13
        ocp.solver_options.qp_solver_iter_max = 50
        ocp.solver_options.tol = 1e-3
        ocp.solver_options.globalization = "MERIT_BACKTRACKING"

        os.environ["ACADOS_SOURCE_DIR"] = str(
            Path(__file__).resolve().parent.parent / "acados"
        )
        return AcadosOcpSolver(ocp)

    def reset_warm_start(self, q_init):
        x0 = np.zeros(self._nx)
        x0[:self._nq] = q_init
        for k in range(self._N + 1):
            self._ocp_solver.set(k, "x", x0)
        for k in range(self._N):
            self._ocp_solver.set(k, "u", np.zeros(self._nu))

    def solve(self, marker_window):
        assert len(marker_window) == self._N

        x0 = self._ocp_solver.get(0, "x")
        self._ocp_solver.constraints_set(0, "lbx", x0)
        self._ocp_solver.constraints_set(0, "ubx", x0)

        for k in range(self._N):
            p = np.zeros(self._nmc)
            for i, key in enumerate(self._valid_keys):
                if key in marker_window[k]:
                    p[3 * i:3 * i + 3] = (
                        np.array(marker_window[k][key]).flatten()[:3]
                    )
            yref = np.concatenate(
                [p, np.zeros(self._nv), np.zeros(self._nu)]
            )
            self._ocp_solver.set(k, "yref", yref)
            self._ocp_solver.set(k, "p", p)

        self._ocp_solver.set(self._N, "yref", np.zeros(self._nv))
        self._ocp_solver.solve()

        x_sol = self._ocp_solver.get(self._N - 1, "x")
        q_sol = x_sol[:self._nq]

        for k in range(self._N):
            self._ocp_solver.set(k, "x", self._ocp_solver.get(k + 1, "x"))
            if k < self._N - 1:
                self._ocp_solver.set(
                    k, "u", self._ocp_solver.get(k + 1, "u"))

        return q_sol


# ═══════════════════════════════════════════════════════════════════════════
# SCALING HELPERS
# ═══════════════════════════════════════════════════════════════════════════

# Augmenter output marker names — must match run_marker_augmenter.py MARKERS order
_AUGMENTED_MARKER_NAMES = [
    "RASI", "LASI", "RPSI", "LPSI",
    "RKNE", "RMKNE", "RANK", "RMANK", "RTOE", "R5MHD", "RHEE",
    "LKNE", "LMKNE", "LANK", "LMANK", "LTOE", "LHEE", "L5MHD",
    "RSHO", "LSHO", "C7",
    "r_thigh1_study", "r_thigh2_study", "r_thigh3_study",
    "L_thigh1_study", "L_thigh2_study", "L_thigh3_study",
    "r_sh1_study", "r_sh2_study", "r_sh3_study",
    "L_sh1_study", "L_sh2_study", "L_sh3_study",
    "RHJC_study", "LHJC_study",
    "RELB", "RMELB", "RWRI", "RMWRI",
    "LELB", "LMELB", "LWRI", "LMWRI",
]

_COSMIK_IDX = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4,
    5: 5, 6: 6, 7: 7, 8: 8,
    9: 62, 10: 41, 11: 9, 12: 10,
    13: 11, 14: 12, 15: 13, 16: 14,
    18: 69, 20: 15, 21: 18, 22: 16, 23: 19, 24: 17, 25: 20,
}


def _goliath_frame_to_cosmik26(goliath_frame: np.ndarray) -> np.ndarray:
    """Convert one (70, 3) Goliath frame to (26, 3) COSMIK format."""
    out = np.zeros((26, 3), dtype=np.float64)
    for ci, gi in _COSMIK_IDX.items():
        out[ci] = goliath_frame[gi]
    out[17] = (goliath_frame[3] + goliath_frame[4]) / 2.0  # Head = mid ears
    out[19] = (goliath_frame[9] + goliath_frame[10]) / 2.0  # midHip
    return out


def scale_model_from_goliath(
    human_model, goliath_data, augmenter_path,
    subject_height, subject_weight, gender,
    buffer_size=30, start_sample=0,
):
    """Scale the pinocchio model in-place using the LSTM augmenter on the first frames."""
    augmenter_path = str(augmenter_path)
    try:
        models = loadModel(augmenterDir=augmenter_path)
    except Exception as e:
        print(f"[WARN] Could not load augmenter ({e}), skipping body scaling")
        return human_model

    # Find valid (non-NaN) frames to build the augmenter buffer
    valid_frames = [
        f for f in goliath_data
        if not np.isnan(f).any()
    ]
    if len(valid_frames) == 0:
        print("[WARN] All frames contain NaN, skipping body scaling")
        return human_model
    # Use up to buffer_size valid frames, padded at start if needed
    frames = np.array(valid_frames[:buffer_size])
    cosmik_buffer = np.array([_goliath_frame_to_cosmik26(f) for f in frames])
    if len(cosmik_buffer) < buffer_size:
        pad = np.tile(cosmik_buffer[0:1], (buffer_size - len(cosmik_buffer), 1, 1))
        cosmik_buffer = np.concatenate([pad, cosmik_buffer], axis=0)

    try:
        augmented = augmentTRC(
            cosmik_buffer, subject_mass=subject_weight,
            subject_height=subject_height, models=models,
            augmenterDir=augmenter_path,
        )
    except Exception as e:
        print(f"[WARN] Augmenter failed ({e}), skipping body scaling")
        return human_model

    # Reshape flat array → marker dict
    mks = {}
    for i, name in enumerate(_AUGMENTED_MARKER_NAMES):
        mks[name] = augmented[i * 3: i * 3 + 3]

    # The augmenter outputs body markers only (no head) → scale_human_model would
    # fall back to the mocap head branch needing FHD/RHD/LHD (KeyError 'FHD').
    # Add COSMIK head markers from the same buffer frame as augmentTRC's output
    # (last frame, same reference frame) so it uses the COSMIK head branch instead.
    head_src = cosmik_buffer[-1]
    mks["Nose"]      = head_src[0]
    mks["Left_Eye"]  = head_src[1]
    mks["Right_Eye"] = head_src[2]
    mks["Left_Ear"]  = head_src[3]
    mks["Right_Ear"] = head_src[4]
    mks["Head"]      = head_src[17]   # COSMIK 17 = mid-ears

    try:
        human_model = scale_human_model(
            human_model, mks,
            with_hand=False, gender=gender,
            subject_height=subject_height,
        )
        print(f"[INFO] Body scaled to {subject_height}m / {subject_weight}kg ({gender})")
    except Exception as e:
        print(f"[WARN] scale_human_model failed ({e}), using unscaled model")

    return human_model


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent


def parse_args():
    p = argparse.ArgumentParser(
        description="SAM3D full-body + MANO hand IK via ACADOS MPC")
    p.add_argument("--id", dest="subject_id", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--output-root",
                   default=_PROJECT_DIR / "output" / "res_hpe", type=Path)
    p.add_argument("--model-dir", default=_PROJECT_DIR / "model", type=Path)
    p.add_argument("--urdf-file", default="urdf/human_with_mano.urdf")
    p.add_argument("--keypoints-file",
                   default="dexsuite_joints_sam3d.npy")
    p.add_argument("--input-scale", type=float, default=1.0,
                   help="Scale factor for input (0.001 for mm->m)")
    p.add_argument("--no-cv-to-ros", action="store_true",
                   help="Skip OpenCV->ROS coordinate conversion")
    p.add_argument("--tilt-deg", type=float, default=0.0,
                   help="Correct camera downward tilt in degrees (e.g. 15 if camera points 15° down)")
    p.add_argument("--dt", type=float, default=0.025)
    p.add_argument("--N", type=int, default=3)
    p.add_argument("--subject-height", type=float, default=1.80,
                   help="Subject height in metres for body scaling (default: 1.80)")
    p.add_argument("--subject-weight", type=float, default=75.0,
                   help="Subject weight in kg for body scaling (default: 75.0)")
    p.add_argument("--gender", default="male", choices=["male", "female"])
    p.add_argument("--augmenter-path", default="augmentation_model",
                   help="Path to LSTM augmenter model directory")
    p.add_argument("--w-markers", type=float, default=1.0)
    p.add_argument("--w-hand", type=float, default=3.0,
                   help="Weight for hand keypoints (higher = faster/tighter hand tracking)")
    p.add_argument("--w-fingertip-mult", type=float, default=2.0,
                   help="Multiplier on w_hand for fingertip keypoints (*4). Default: 2.0")
    p.add_argument("--w-dq", type=float, default=1e-5)
    p.add_argument("--w-dq-hand", type=float, default=None,
                   help="Velocity regularization for MANO finger joints (default: same as --w-dq). "
                        "Set lower (e.g. 1e-7) to allow faster hand closing.")
    p.add_argument("--w-u", type=float, default=1e-6)
    p.add_argument("--start-sample", type=int, default=0)
    p.add_argument("--display", action="store_true")
    p.add_argument("--save-video", action="store_true")
    p.add_argument("--meshcat-url", default="tcp://127.0.0.1:6000")
    p.add_argument("--output-angles-file",
                   default="joint_angles_sam3d_mpc.csv")
    p.add_argument("--output-markers-file",
                   default="markers_model_sam3d_mpc.csv")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    model_dir = Path(args.model_dir).resolve()
    output_root = Path(args.output_root).resolve()
    urdf_path = str(model_dir / args.urdf_file)

    task_dir = output_root / args.subject_id / args.task
    input_path = task_dir / args.keypoints_file
    output_dir = task_dir
    output_angles_csv = task_dir / args.output_angles_file
    output_markers_csv = task_dir / args.output_markers_file

    if not Path(urdf_path).exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    print(f"[INFO] Subject {args.subject_id}, Task {args.task}")

    # ── 1. Load SAM3D data ────────────────────────────────────────────────
    print(f"[INFO] Loading SAM3D data from {input_path}")
    goliath_data = load_sam3d_data(
        input_path, scale=args.input_scale,
        cv_to_ros=not args.no_cv_to_ros,
        tilt_deg=args.tilt_deg,
    )
    n_frames = goliath_data.shape[0]
    print(f"[INFO] {n_frames} frames, {goliath_data.shape[1]} keypoints")

    result_markers = []
    for t in range(n_frames):
        frame_dict = {}
        for ki, name in enumerate(GOLIATH_NAMES):
            frame_dict[name] = goliath_data[t, ki]
        result_markers.append(frame_dict)

    # ── 2. Load URDF ────────────────────────────────────────────────────
    print(f"[INFO] Loading URDF: {urdf_path}")
    urdf_dir = str(Path(urdf_path).parent)
    human_model, human_coll, human_vis = pin.buildModelsFromUrdf(
        urdf_path,
        urdf_dir,
        root_joint=pin.JointModelFreeFlyer(),
    )
    print(f"[INFO] Raw model: nq={human_model.nq}, nv={human_model.nv}")

    # ── 2b. Scale model to subject dimensions ────────────────────────────
    augmenter_path = Path(args.augmenter_path)
    if not augmenter_path.is_absolute():
        augmenter_path = _PROJECT_DIR / augmenter_path
    human_model = scale_model_from_goliath(
        human_model, goliath_data,
        augmenter_path=augmenter_path,
        subject_height=args.subject_height,
        subject_weight=args.subject_weight,
        gender=args.gender,
        start_sample=args.start_sample,
    )

    # ── 3. Register tracking frames ──────────────────────────────────────
    human_model, registered = register_tracking_frames(human_model)
    print(f"[INFO] Registered {len(registered)} tracking frames")

    # ── 4. Lock thoracic joints (wrists UNLOCKED for MANO coupling) ──────
    joint_ids = [
        human_model.getJointId(j)
        for j in JOINTS_TO_LOCK
        if human_model.existJointName(j)
    ]
    q0 = pin.neutral(human_model)
    if joint_ids:
        human_model, (human_coll, human_vis) = pin.buildReducedModel(
            human_model, [human_coll, human_vis], joint_ids, q0,
        )
        print(f"[INFO] Locked {len(joint_ids)} joints -> "
              f"nq={human_model.nq}, nv={human_model.nv}")

    human_data = pin.Data(human_model)

    valid_solver_keys = [
        k for k in SOLVER_KEYS if human_model.existFrame(k)
    ]
    print(f"[INFO] Solver will track {len(valid_solver_keys)} markers "
          f"({len(BODY_SOLVER_KEYS)} body + "
          f"{len(valid_solver_keys) - len(BODY_SOLVER_KEYS)} hand)")

    # ── 5. Visualization (optional) ──────────────────────────────────────
    viz = None
    viewer = None
    if args.display:
        import meshcat
        import meshcat.geometry
        import meshcat.transformations
        import webbrowser
        from pinocchio.visualize import MeshcatVisualizer

        viz = MeshcatVisualizer(human_model, human_coll, human_vis)
        viz.initViewer(open=False)
        viewer = viz.viewer
        viz.loadViewerModel("human_model")

        viewer["/Background"].set_property("top_color", [1, 1, 1])
        viewer["/Background"].set_property("bottom_color", [0.65, 0.65, 0.65])

        grid_height = -1.0
        viewer["/Grid"].set_transform(np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, grid_height],
            [0, 0, 0, 1],
        ]))

        for mk in valid_solver_keys:
            viewer[f"gt/{mk}"].set_object(
                meshcat.geometry.Sphere(0.012),
                meshcat.geometry.MeshLambertMaterial(
                    color=0xFF0000, opacity=0.5),
            )
            viewer[f"est/{mk}"].set_object(
                meshcat.geometry.Sphere(0.008),
                meshcat.geometry.MeshLambertMaterial(
                    color=0x000000, opacity=0.5),
            )

        viz.display(pin.neutral(human_model))
        first_mks = result_markers[max(0, args.start_sample)]
        for mk in valid_solver_keys:
            if mk in first_mks:
                pos = np.array(first_mks[mk]).flatten()[:3]
                viewer[f"gt/{mk}"].set_transform(
                    meshcat.transformations.translation_matrix(pos))

        url = viewer.url()
        print(f"[INFO] Open visualizer at: {url}")
        webbrowser.open(url, new=2)
        print("[INFO] Meshcat ready -- press Enter to start IK")
        input()

    # ── 6. IPOPT warm-start (first frame) ────────────────────────────────
    print("[INFO] IPOPT warm-start (first frame)...")
    from comfi_examples.ik_utils import RT_IK

    warmstart_keys = [k for k in registered if human_model.existFrame(k)]
    omega = {}
    for k in warmstart_keys:
        omega[k] = 0.5 if k in HAND_KP_TO_MANO_LINK else 1.0

    q0 = pin.neutral(human_model)
    rt_ik = RT_IK(
        human_model, result_markers[args.start_sample], q0,
        warmstart_keys, args.dt, omega,
    )
    q_init = rt_ik.solve_ik_sample_casadi()
    print("[INFO] IPOPT warm-start done")

    # ── 7. Create MPC solver ─────────────────────────────────────────────
    print(f"[INFO] Building ACADOS solver (N={args.N}, dt={args.dt}, "
          f"{len(valid_solver_keys)} markers)...")
    ik_solver = AcadosMPCIKSolver(
        pin_model=human_model,
        keys_to_track=valid_solver_keys,
        N=args.N,
        dt=args.dt,
        w_markers=args.w_markers,
        w_hand=args.w_hand,
        w_dq=args.w_dq,
        w_dq_hand=args.w_dq_hand,
        w_u=args.w_u,
        hand_keys=HAND_SOLVER_KEYS,
        w_fingertip_mult=args.w_fingertip_mult,
    )
    ik_solver.reset_warm_start(q_init)

    # ── 8. MPC loop ──────────────────────────────────────────────────────
    print(f"[INFO] Solving frames [{args.start_sample}, {n_frames})...")
    q_list = []
    M_model_list = []
    rmse_per_marker = {}
    solve_times = []
    video_frames = [] if (args.display and args.save_video) else None

    for ii in range(args.start_sample, n_frames):
        window = [
            result_markers[max(args.start_sample, ii - args.N + 1 + k)]
            for k in range(args.N)
        ]

        t0 = time.perf_counter()
        q = ik_solver.solve(window)
        solve_times.append(time.perf_counter() - t0)

        pin.forwardKinematics(human_model, human_data, q)
        pin.updateFramePlacements(human_model, human_data)

        cur = window[-1]
        M_frame = {}
        for mk in valid_solver_keys:
            if mk not in cur or not human_model.existFrame(mk):
                continue
            fid = human_model.getFrameId(mk)
            gt = np.array(cur[mk]).flatten()[:3]
            est = np.array(human_data.oMf[fid].translation).flatten()

            M_frame[f"{mk}_x"] = est[0]
            M_frame[f"{mk}_y"] = est[1]
            M_frame[f"{mk}_z"] = est[2]

            rmse_per_marker.setdefault(mk, []).append(
                np.sum((gt - est) ** 2))

            if viewer:
                import meshcat.transformations
                viewer[f"gt/{mk}"].set_transform(
                    meshcat.transformations.translation_matrix(gt))
                viewer[f"est/{mk}"].set_transform(
                    meshcat.transformations.translation_matrix(est))

        M_model_list.append(M_frame)
        q_list.append(q)

        if viz:
            viz.display(q)
        if video_frames is not None:
            video_frames.append(viz.captureImage())
        if ii % 10 == 0:
            print(f"  frame {ii}/{n_frames}")

    print("[INFO] MPC complete")

    # ── 9. Save results ──────────────────────────────────────────────────
    task_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(q_list).to_csv(output_angles_csv, index=False)
    pd.DataFrame(M_model_list).to_csv(output_markers_csv, index=False)

    st = np.array(solve_times)
    output_times_csv = task_dir / "solve_times_sam3d_mpc.csv"
    pd.DataFrame({
        "frame": np.arange(args.start_sample, args.start_sample + len(st)),
        "solve_time_s": st,
    }).to_csv(output_times_csv, index=False)

    if video_frames:
        import imageio
        vpath = task_dir / "ik_sam3d_mpc.mp4"
        imageio.mimsave(str(vpath), video_frames, fps=int(1 / args.dt))
        print(f"[INFO] Video saved: {vpath}")

    # ── 10. Report ───────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"SOLVE TIME  (N={args.N})")
    print(f"  mean  {st.mean() * 1e3:.1f} ms   "
          f"median {np.median(st) * 1e3:.1f} ms   "
          f"max {st.max() * 1e3:.1f} ms")
    print(f"{'=' * 60}")
    print("RMSE per marker (m):")
    total, cnt = 0.0, 0
    for mk, errs in sorted(rmse_per_marker.items()):
        r = np.sqrt(np.mean(errs))
        total += r
        cnt += 1
        print(f"  {mk:45s} {r:.4f}")
    if cnt:
        print(f"  {'GLOBAL':45s} {total / cnt:.4f}")
    print(f"{'=' * 60}")

    print(f"\n[SUCCESS] {args.subject_id}/{args.task} SAM3D IK complete")
    print(f"  Joint angles : {output_angles_csv}")
    print(f"  Model markers: {output_markers_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
