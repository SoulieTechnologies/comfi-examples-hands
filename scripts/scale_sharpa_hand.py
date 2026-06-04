#!/usr/bin/env python3
"""
Scale Sharpa Hand URDF locally per-link to match human hand bone ratios
from sam3D 21-joint keypoints.

Approach (inspired by RT-COSMIK model_utils.py):
  1. Load the sam3D 21-joint keypoints for one hand
  2. Compute human bone lengths (Euclidean distance between consecutive joints)
  3. For each corresponding URDF joint, read the original translation
  4. Compute a per-link scale factor = human_length / robot_length
  5. Scale only the dominant translation axis of each joint origin in the URDF XML
  6. Write the scaled URDF and optionally validate with pinocchio FK

Usage:
  python scale_sharpa_hand.py \
    --urdf path/to/right_sharpa_wave_with_wrist.urdf \
    --keypoints path/to/sam3d_hand_keypoints.npy \
    --output path/to/right_sharpa_wave_scaled.urdf \
    [--hand-index 1] \
    [--frame-index 0] \
    [--validate]
"""

import argparse
import copy
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# sam3D 21-joint hand skeleton (tip→base ordering, same as MHR/MANO)
# ──────────────────────────────────────────────────────────────────────────────
JOINT_NAMES = [
    "THUMB_TIP",    # 0
    "THUMB_DIP",    # 1
    "THUMB_PIP",    # 2
    "THUMB_MCP",    # 3
    "INDEX_TIP",    # 4
    "INDEX_DIP",    # 5
    "INDEX_PIP",    # 6
    "INDEX_MCP",    # 7
    "MIDDLE_TIP",   # 8
    "MIDDLE_DIP",   # 9
    "MIDDLE_PIP",   # 10
    "MIDDLE_MCP",   # 11
    "RING_TIP",     # 12
    "RING_DIP",     # 13
    "RING_PIP",     # 14
    "RING_MCP",     # 15
    "PINKY_TIP",    # 16
    "PINKY_DIP",    # 17
    "PINKY_PIP",    # 18
    "PINKY_MCP",    # 19
    "WRIST",        # 20
]

# ──────────────────────────────────────────────────────────────────────────────
# Mapping: which human bone segments to compute, and which URDF joint
# encodes that bone length.
#
# Each entry is:
#   (sam3d_idx_proximal, sam3d_idx_distal, urdf_joint_name, dominant_axis_index)
#
# The dominant_axis_index is 0=x, 1=y, 2=z — this is the component of the
# joint <origin xyz="..."> that encodes the bone length.
# ──────────────────────────────────────────────────────────────────────────────

# For multi-axis joints (like the thumb CMC), we scale the overall translation
# magnitude while preserving the direction vector.

BONE_SEGMENTS = {
    "thumb": [
        # CMC → MCP: joint "right_thumb_MCP_FE", dominant axis x (0.065)
        (3, 2, "right_thumb_MCP_FE", "x"),
        # MCP → IP: joint "right_thumb_IP", dominant axis x (0.039)
        (2, 1, "right_thumb_IP", "x"),
        # IP → Tip: through elastomer fixed chain, the total DP→fingertip
        # distance is encoded across right_thumb_IP output → DP → elastomer → fingertip
        # We scale the fingertip_fix_joint dominant axis y (-0.027611)
        (1, 0, "right_thumb_fingertip_fix_joint", "y"),
    ],
    "index": [
        # MCP → PIP: joint "right_index_PIP", dominant axis x (0.047)
        (7, 6, "right_index_PIP", "x"),
        # PIP → DIP: joint "right_index_DIP", dominant axis x (0.0315)
        (6, 5, "right_index_DIP", "x"),
        # DIP → Tip: fingertip_fix_joint, dominant axis y (-0.026009)
        (5, 4, "right_index_fingertip_fix_joint", "y"),
    ],
    "middle": [
        (11, 10, "right_middle_PIP", "x"),
        (10, 9, "right_middle_DIP", "x"),
        (9, 8, "right_middle_fingertip_fix_joint", "y"),
    ],
    "ring": [
        (15, 14, "right_ring_PIP", "x"),
        (14, 13, "right_ring_DIP", "x"),
        (13, 12, "right_ring_fingertip_fix_joint", "y"),
    ],
    "pinky": [
        (19, 18, "right_pinky_PIP", "x"),
        (18, 17, "right_pinky_DIP", "x"),
        (17, 16, "right_pinky_fingertip_fix_joint", "y"),
    ],
}

# Wrist → MCP segments (scaled on the dominant axis of the MCP_FE joint)
WRIST_TO_MCP_SEGMENTS = {
    "index": (20, 7, "right_index_MCP_FE", "z"),      # z=0.0959
    "middle": (20, 11, "right_middle_MCP_FE", "z"),    # z=0.0989
    "ring": (20, 15, "right_ring_MCP_FE", "z"),        # z=0.0929
    # Pinky has a more complex chain: CMC + MCP_FE
    # We'll scale the CMC joint's dominant axis instead
    "pinky": (20, 19, "right_pinky_CMC", "z"),         # z=0.0869 (in the xyz)
}

# Thumb wrist→CMC: multi-axis translation on right_thumb_CMC_FE
THUMB_WRIST_TO_CMC = (20, 3, "right_thumb_CMC_FE", "magnitude")

AXIS_MAP = {"x": 0, "y": 1, "z": 2}


# ──────────────────────────────────────────────────────────────────────────────
# Keypoint loading
# ──────────────────────────────────────────────────────────────────────────────

def load_keypoints(path: str, hand_index: int = 1, frame_index: int = 0) -> np.ndarray:
    """
    Load sam3D keypoints and return a (21, 3) array for a single hand frame.

    Supports:
      - Shape (N, 2, 21, 3): multi-frame, two-hand output from sam_extractor.py
      - Shape (21, 3): single-frame, single-hand calibration
      - Shape (N, 21, 3): multi-frame, single-hand
    """
    data = np.load(path)
    if data.ndim == 4:
        # (N_frames, 2_hands, 21_joints, 3_xyz)
        kp = data[frame_index, hand_index, :, :]
    elif data.ndim == 3:
        # (N_frames, 21_joints, 3_xyz)
        kp = data[frame_index, :, :]
    elif data.ndim == 2:
        # (21_joints, 3_xyz)
        kp = data
    else:
        raise ValueError(f"Unexpected keypoints shape: {data.shape}")

    assert kp.shape == (21, 3), f"Expected (21, 3), got {kp.shape}"
    return kp


# ──────────────────────────────────────────────────────────────────────────────
# Bone length computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_human_bone_lengths(keypoints: np.ndarray) -> dict:
    """
    Compute human bone lengths from 21-joint keypoints.

    Returns a dict: {urdf_joint_name: human_length_in_meters}
    """
    lengths = {}

    # Finger phalanges
    for finger_name, segments in BONE_SEGMENTS.items():
        for prox_idx, dist_idx, joint_name, _ in segments:
            length = np.linalg.norm(keypoints[prox_idx] - keypoints[dist_idx])
            lengths[joint_name] = length

    # Wrist → MCP
    for finger_name, (prox_idx, dist_idx, joint_name, _) in WRIST_TO_MCP_SEGMENTS.items():
        length = np.linalg.norm(keypoints[prox_idx] - keypoints[dist_idx])
        lengths[joint_name] = length

    # Thumb wrist → CMC
    prox_idx, dist_idx, joint_name, _ = THUMB_WRIST_TO_CMC
    length = np.linalg.norm(keypoints[prox_idx] - keypoints[dist_idx])
    lengths[joint_name] = length

    return lengths


# ──────────────────────────────────────────────────────────────────────────────
# Robot bone length extraction from URDF
# ──────────────────────────────────────────────────────────────────────────────

def get_joint_origin(root: ET.Element, joint_name: str):
    """Find a joint element by name and return its (x, y, z) origin."""
    for joint in root.findall("joint"):
        if joint.attrib.get("name") == joint_name:
            origin = joint.find("origin")
            if origin is not None and "xyz" in origin.attrib:
                return [float(v) for v in origin.attrib["xyz"].split()]
    return None


def compute_robot_bone_lengths(root: ET.Element) -> dict:
    """
    Extract robot bone lengths from the URDF joint origins.

    For single-axis joints, returns the absolute value of the dominant axis.
    For magnitude joints (thumb CMC), returns the Euclidean norm of the translation.
    """
    lengths = {}

    # Finger phalanges
    for finger_name, segments in BONE_SEGMENTS.items():
        for _, _, joint_name, axis in segments:
            xyz = get_joint_origin(root, joint_name)
            if xyz is None:
                print(f"  [WARN] Joint '{joint_name}' not found in URDF")
                continue
            if axis == "magnitude":
                lengths[joint_name] = np.linalg.norm(xyz)
            else:
                lengths[joint_name] = abs(xyz[AXIS_MAP[axis]])

    # Wrist → MCP
    for finger_name, (_, _, joint_name, axis) in WRIST_TO_MCP_SEGMENTS.items():
        xyz = get_joint_origin(root, joint_name)
        if xyz is None:
            print(f"  [WARN] Joint '{joint_name}' not found in URDF")
            continue
        lengths[joint_name] = abs(xyz[AXIS_MAP[axis]])

    # Thumb wrist → CMC (magnitude)
    _, _, joint_name, mode = THUMB_WRIST_TO_CMC
    xyz = get_joint_origin(root, joint_name)
    if xyz is not None:
        lengths[joint_name] = np.linalg.norm(xyz)
    else:
        print(f"  [WARN] Joint '{joint_name}' not found in URDF")

    return lengths


# ──────────────────────────────────────────────────────────────────────────────
# Scale factor computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_scale_factors(human_lengths: dict, robot_lengths: dict) -> dict:
    """
    Compute per-joint scale factors: s = human_length / robot_length.
    """
    factors = {}
    for joint_name in human_lengths:
        if joint_name not in robot_lengths:
            continue
        robot_l = robot_lengths[joint_name]
        human_l = human_lengths[joint_name]
        if robot_l < 1e-6:
            continue
        factors[joint_name] = human_l / robot_l
    return factors


# ──────────────────────────────────────────────────────────────────────────────
# Apply scales to URDF
# ──────────────────────────────────────────────────────────────────────────────

def apply_scales_to_urdf(root: ET.Element, scale_factors: dict):
    """
    Modify joint origin translations in-place according to the scale factors.

    For single-axis scaling: only the dominant axis component is multiplied.
    For magnitude scaling: all three components are multiplied by the same ratio.
    """
    # Build a lookup: joint_name → (axis_mode, factor)
    axis_lookup = {}

    for finger_name, segments in BONE_SEGMENTS.items():
        for _, _, joint_name, axis in segments:
            if joint_name in scale_factors:
                axis_lookup[joint_name] = (axis, scale_factors[joint_name])

    for finger_name, (_, _, joint_name, axis) in WRIST_TO_MCP_SEGMENTS.items():
        if joint_name in scale_factors:
            axis_lookup[joint_name] = (axis, scale_factors[joint_name])

    _, _, joint_name, mode = THUMB_WRIST_TO_CMC
    if joint_name in scale_factors:
        axis_lookup[joint_name] = (mode, scale_factors[joint_name])

    # Apply
    for joint in root.findall("joint"):
        jname = joint.attrib.get("name")
        if jname not in axis_lookup:
            continue

        origin = joint.find("origin")
        if origin is None or "xyz" not in origin.attrib:
            continue

        xyz = [float(v) for v in origin.attrib["xyz"].split()]
        axis_mode, factor = axis_lookup[jname]

        if axis_mode == "magnitude":
            # Scale all components uniformly to preserve direction
            xyz = [v * factor for v in xyz]
        else:
            idx = AXIS_MAP[axis_mode]
            xyz[idx] *= factor

        origin.attrib["xyz"] = f"{xyz[0]:.6g} {xyz[1]:.6g} {xyz[2]:.6g}"


# ──────────────────────────────────────────────────────────────────────────────
# Pinocchio FK validation
# ──────────────────────────────────────────────────────────────────────────────

def validate_with_pinocchio(urdf_path: str, keypoints: np.ndarray, package_dirs: str):
    """
    Load the scaled URDF with pinocchio, run FK at neutral q,
    and compare fingertip-to-wrist distances with human keypoints.
    """
    try:
        import pinocchio as pin
    except ImportError:
        print("[WARN] pinocchio not available, skipping FK validation")
        return

    # Sharpa frame names for the 21 sam3D joints
    frame_mapping = {
        0: "right_thumb_fingertip",
        1: "right_thumb_DP",
        2: "right_thumb_PP",
        3: "right_thumb_MC",
        4: "right_index_fingertip",
        5: "right_index_DP",
        6: "right_index_MP",
        7: "right_index_PP",
        8: "right_middle_fingertip",
        9: "right_middle_DP",
        10: "right_middle_MP",
        11: "right_middle_PP",
        12: "right_ring_fingertip",
        13: "right_ring_DP",
        14: "right_ring_MP",
        15: "right_ring_PP",
        16: "right_pinky_fingertip",
        17: "right_pinky_DP",
        18: "right_pinky_MP",
        19: "right_pinky_PP",
    }

    model = pin.buildModelFromUrdf(urdf_path, package_dirs, pin.JointModelFreeFlyer())
    data = model.createData()

    q = pin.neutral(model)
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    # Center keypoints on wrist
    wrist_pos = keypoints[20].copy()
    kp_centered = keypoints - wrist_pos

    print("\n" + "=" * 80)
    print("PINOCCHIO FK VALIDATION (neutral q)")
    print("=" * 80)

    # Compute per-finger bone length comparisons
    fingers = {
        "Thumb": [(3, 2), (2, 1), (1, 0)],
        "Index": [(7, 6), (6, 5), (5, 4)],
        "Middle": [(11, 10), (10, 9), (9, 8)],
        "Ring": [(15, 14), (14, 13), (13, 12)],
        "Pinky": [(19, 18), (18, 17), (17, 16)],
    }

    bone_names = {
        (3, 2): "MC→PP",
        (2, 1): "PP→DP",
        (1, 0): "DP→Tip",
        (7, 6): "PP→MP",
        (6, 5): "MP→DP",
        (5, 4): "DP→Tip",
        (11, 10): "PP→MP",
        (10, 9): "MP→DP",
        (9, 8): "DP→Tip",
        (15, 14): "PP→MP",
        (14, 13): "MP→DP",
        (13, 12): "DP→Tip",
        (19, 18): "PP→MP",
        (18, 17): "MP→DP",
        (17, 16): "DP→Tip",
    }

    print(
        f"{'Finger':<8} {'Bone':<10} {'Human (mm)':>12} {'Robot (mm)':>12} {'Error (mm)':>12}"
    )
    print("-" * 56)

    for finger, pairs in fingers.items():
        for prox_idx, dist_idx in pairs:
            # Human
            human_l = np.linalg.norm(kp_centered[prox_idx] - kp_centered[dist_idx]) * 1000

            # Robot FK
            if prox_idx in frame_mapping and dist_idx in frame_mapping:
                prox_fid = model.getFrameId(frame_mapping[prox_idx])
                dist_fid = model.getFrameId(frame_mapping[dist_idx])
                robot_prox = data.oMf[prox_fid].translation
                robot_dist = data.oMf[dist_fid].translation
                robot_l = np.linalg.norm(robot_prox - robot_dist) * 1000
            else:
                robot_l = float("nan")

            err = human_l - robot_l
            bname = bone_names.get((prox_idx, dist_idx), "?")
            print(
                f"{finger:<8} {bname:<10} {human_l:>12.2f} {robot_l:>12.2f} {err:>+12.2f}"
            )

    print("=" * 80)


# ──────────────────────────────────────────────────────────────────────────────
# Pretty-print report
# ──────────────────────────────────────────────────────────────────────────────

def print_scaling_report(
    human_lengths: dict, robot_lengths: dict, scale_factors: dict
):
    """Print a nice table of all bone segments with lengths and scale factors."""
    print("\n" + "=" * 80)
    print("SHARPA HAND LOCAL SCALING REPORT")
    print("=" * 80)
    print(
        f"{'Joint Name':<40} {'Human (mm)':>12} {'Robot (mm)':>12} {'Scale':>8}"
    )
    print("-" * 74)

    for jname in sorted(human_lengths.keys()):
        h = human_lengths[jname] * 1000  # m → mm
        r = robot_lengths.get(jname, 0) * 1000
        s = scale_factors.get(jname, float("nan"))
        print(f"{jname:<40} {h:>12.2f} {r:>12.2f} {s:>8.4f}")

    print("=" * 80)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Locally scale Sharpa Hand URDF per-link to match human hand bone ratios"
    )
    parser.add_argument(
        "--urdf",
        required=True,
        help="Path to the input Sharpa Hand URDF (with wrist)",
    )
    parser.add_argument(
        "--keypoints",
        required=True,
        help="Path to sam3D keypoints .npy file",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path for the output scaled URDF",
    )
    parser.add_argument(
        "--hand-index",
        type=int,
        default=1,
        help="Hand index in the (N,2,21,3) array: 0=left, 1=right (default: 1)",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="Frame index to use for calibration (default: 0)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run pinocchio FK validation after scaling",
    )
    parser.add_argument(
        "--package-dirs",
        default=None,
        help="Package directory for pinocchio URDF loading (for validation). "
             "Defaults to the parent directory of the output URDF.",
    )
    args = parser.parse_args()

    # ── 1. Load keypoints ──
    print(f"[1/5] Loading keypoints from: {args.keypoints}")
    keypoints = load_keypoints(args.keypoints, args.hand_index, args.frame_index)
    print(f"       Keypoint shape: {keypoints.shape}")

    # ── 2. Compute human bone lengths ──
    print("[2/5] Computing human bone lengths...")
    human_lengths = compute_human_bone_lengths(keypoints)

    # ── 3. Parse URDF and compute robot bone lengths ──
    print(f"[3/5] Parsing URDF: {args.urdf}")
    tree = ET.parse(args.urdf)
    root = tree.getroot()
    robot_lengths = compute_robot_bone_lengths(root)

    # ── 4. Compute scale factors and apply ──
    print("[4/5] Computing per-link scale factors...")
    scale_factors = compute_scale_factors(human_lengths, robot_lengths)
    print_scaling_report(human_lengths, robot_lengths, scale_factors)

    print("[5/5] Applying scales to URDF...")
    apply_scales_to_urdf(root, scale_factors)

    # ── Write output ──
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    tree.write(args.output, xml_declaration=True, encoding="utf-8")
    print(f"\n[DONE] Scaled URDF saved to: {args.output}")

    # ── Validate ──
    if args.validate:
        pkg = args.package_dirs or str(Path(args.output).parent)
        validate_with_pinocchio(args.output, keypoints, pkg)


if __name__ == "__main__":
    sys.exit(main())
