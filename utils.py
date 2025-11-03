import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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


def generate_plot_path(
    base_path, motion_id, joint_idx, joint_name, save_plot_name, suffix
):
    folder_path = os.path.join(base_path, str(motion_id))
    os.makedirs(folder_path, exist_ok=True)
    return os.path.join(
        folder_path, f"{motion_id}_{joint_idx+1}{joint_name}_{save_plot_name}_{suffix}.png"
    )


def plot_joint_trajectory(x, y, z, joint_idx, joint_name, save_name, min_vals, max_vals, motion_id):
    x_min, x_max = round(float(min_vals[2]), 2), round(float(max_vals[2]), 2)  # X
    y_min, y_max = round(float(min_vals[0]), 2), round(float(max_vals[0]), 2)  # Y
    z_min, z_max = round(float(min_vals[1]), 2), round(float(max_vals[1]), 2)  # Z
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(x, y, z, c=z, alpha=0.7, depthshade=True)

    if x and y and z:
        ax.plot(x, y, "+", markersize=1.0, color="r", label="XY")
        ax.plot(x, z, "+", markersize=1.0, color="g", label="XZ")
        ax.plot(y, z, "+", markersize=1.0, color="b", label="YZ")
        for i in range(0, len(x), 10):
            ax.text(
                x[i], y[i], z[i], f"{i}=={x[i]:.0f},{y[i]:.0f},{z[i]:.0f}", fontsize=2
            )

    ax.set_xlabel("X (flx, ext)", fontsize=14, labelpad=11)
    ax.set_ylabel("Y (int, ext rot)", fontsize=14, labelpad=13)
    ax.set_zlabel("Z (abb, abd)", fontsize=14, labelpad=11)

    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])
    ax.set_xticks(np.linspace(x_min, x_max, 5))
    ax.set_yticks(np.linspace(y_min, y_max, 5))
    ax.set_zticks(np.linspace(z_min, z_max, 5))
    ax.tick_params(axis='x', pad=1, labelsize=7, labelrotation=20)
    ax.tick_params(axis='y',pad=3, labelsize=7)
    ax.tick_params(axis='z', pad=1, labelsize=7, labelrotation=20)
    ax.autoscale(False)      # 자동 스케일 끔

    plt.title(f"{motion_id}_{joint_name}_{save_name}")
    plt.colorbar(sc, pad=0.15)
    plt.savefig(
        generate_plot_path("Results", motion_id, joint_idx, joint_name, save_name, "1"),
        dpi=300,
    )
    plt.clf()
    plt.close(fig)
    
    # 2D subplots - 각 서브플롯에 올바른 축 범위 적용
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))

    # XY
    axes[0].plot(x, y, "+", color="r", markersize=3)
    axes[0].set_xlabel("X (flx, ext)")
    axes[0].set_ylabel("Y")
    axes[0].set_xlim(x_min, x_max)
    axes[0].set_ylim(y_min, y_max)
    axes[0].set_xticks(np.linspace(x_min, x_max, 5))
    axes[0].set_yticks(np.linspace(y_min, y_max, 5))
    axes[0].set_title("XY Plane")
    axes[0].grid(True, alpha=0.3)

    axes[0].scatter(x[0], y[0], s=70, marker='o',
           linewidths=1)
    axes[0].scatter(x[-1], y[-1], s=70, marker='X',
            linewidths=1)

    # XZ
    axes[1].plot(x, z, "+", color="g", markersize=3)
    axes[1].set_xlabel("X (flx, ext)")
    axes[1].set_ylabel("Z (abb, abd)")
    axes[1].set_xlim(x_min, x_max)
    axes[1].set_ylim(z_min, z_max)
    axes[1].set_xticks(np.linspace(x_min, x_max, 5))
    axes[1].set_yticks(np.linspace(z_min, z_max, 5))
    axes[1].set_title("XZ Plane")
    axes[1].grid(True, alpha=0.3)

    axes[1].scatter(x[0], z[0], s=70, marker='o',
           linewidths=1)
    axes[1].scatter(x[-1], z[-1], s=70, marker='X',
            linewidths=1)

    # YZ
    axes[2].plot(y, z, "+", color="b", markersize=3)
    axes[2].set_xlabel("Y")
    axes[2].set_ylabel("Z (abb, abd)")
    axes[2].set_xlim(y_min, y_max)
    axes[2].set_ylim(z_min, z_max)
    axes[2].set_xticks(np.linspace(y_min, y_max, 5))
    axes[2].set_yticks(np.linspace(z_min, z_max, 5))
    axes[2].set_title("YZ Plane")
    axes[2].grid(True, alpha=0.3)

    axes[2].scatter(y[0], z[0], s=70, marker='o',
           linewidths=1)
    axes[2].scatter(y[-1], z[-1], s=70, marker='X',
            linewidths=1)

    plt.suptitle(f"{joint_idx+1}. {joint_name}", fontsize=12)
    plt.tight_layout()
    plt.savefig(
        generate_plot_path("Results", motion_id, joint_idx, joint_name, save_name, "2"),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()


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
        j: np.array([np.inf, np.inf, np.inf], dtype=float)
        for j in lower_body_joints.keys()
        }
global_joint_max = {
        j: np.array([-np.inf, -np.inf, -np.inf], dtype=float)
        for j in lower_body_joints.keys()
        }
