# anglespace의 전체적인 유틸을 담당
import torch
import numpy as np


def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def _index_from_letter(letter: str) -> int:
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2
    raise ValueError("letter must be either X, Y or Z.")


def _angle_from_tan(
    axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
) -> torch.Tensor:
    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])


def matrix_to_euler_angles(matrix: torch.Tensor, convention: str) -> torch.Tensor:
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(
            matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        central_angle = torch.acos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(
            convention[0], convention[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
        ),
    )
    return torch.stack(o, -1)


def convert_pose_to_euler(pose_vec):
    """Axis-angle을 Euler angle로 변환 (도 단위)"""
    q = axis_angle_to_quaternion(torch.Tensor(pose_vec))
    rotmat = quaternion_to_matrix(q)
    euler_rad = matrix_to_euler_angles(rotmat, "YZX")
    return np.rad2deg(euler_rad.numpy())


def check_label_in_annotations(data_entry, target_label):
    """
    frame_ann 또는 seq_anns에서 특정 라벨이 있는지 확인하는 헬퍼 함수
    """
    # frame_ann 체크
    if data_entry.get("frame_ann") is not None:
        labels_list = data_entry["frame_ann"].get("labels", [])
        for label_data in labels_list:
            act_cat_list = label_data.get("act_cat") or []
            for act_cat in act_cat_list:
                if act_cat and target_label in act_cat:
                    return True

    # seq_anns 체크
    if data_entry.get("seq_anns") is not None:
        seq_anns = data_entry["seq_anns"]
        for seq_ann in seq_anns:
            labels_list = seq_ann.get("labels", [])
            for label_data in labels_list:
                act_cat_list = label_data.get("act_cat") or []
                for act_cat in act_cat_list:
                    if act_cat and target_label in act_cat:
                        return True

    return False


# 하체 관절 정보
lower_body_joints = {
    0: "Pelvis",
    1: "L_Hip",
    2: "R_Hip",
    4: "L_Knee",
    5: "R_Knee",
    7: "L_Ankle",
    8: "R_Ankle",
}


# 글로벌 min/max 초기화 (Y,Z,X 순서)
global_joint_min = {
    j: np.array([np.inf, np.inf, np.inf], dtype=float) for j in lower_body_joints.keys()
}
global_joint_max = {
    j: np.array([-np.inf, -np.inf, -np.inf], dtype=float)
    for j in lower_body_joints.keys()
}
