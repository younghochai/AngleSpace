import os
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator, MultipleLocator


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


def plot_joint_trajectory(x, y, z, joint_idx, joint_name, save_name, min_vals, max_vals, motion_id, tick_interval=None):
    # 3축 전체에서 공통 min / max 찾기
    raw_min = float(np.min(min_vals))
    raw_max = float(np.max(max_vals))

    global_min = math.floor(raw_min)   # 최소는 바닥으로
    global_max = math.ceil(raw_max)    # 최대는 천장으로

    # 0대칭 쓰고 싶으면 아래로 대체
    # max_abs = math.ceil(max(abs(raw_min), abs(raw_max)))
    # global_min, global_max = -max_abs, max_abs

    x_min, x_max = int(global_min), int(global_max)
    y_min, y_max = int(global_min), int(global_max)
    z_min, z_max = int(global_min), int(global_max)

    valid = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(z)
    end_idx = np.flatnonzero(valid)[-1] if np.any(valid) else -1
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(x, y, z, c=z, alpha=0.7, depthshade=True)

    if x and y and z:
        ax.plot(x, y, "+", markersize=1.0, color="r", label="XY")
        ax.plot(x, z, "+", markersize=1.0, color="g", label="XZ")
        ax.plot(y, z, "+", markersize=1.0, color="b", label="YZ")
        for i in range(0, len(x), 30):
            ax.text(
                x[i], y[i], z[i], f"{i}=={x[i]:.0f},{y[i]:.0f},{z[i]:.0f}", fontsize=2
            )

    ax.set_xlabel("X (flx, ext)", fontsize=14, labelpad=11)
    ax.set_ylabel("Y (int, ext rot)", fontsize=14, labelpad=13)
    ax.set_zlabel("Z (abb, abd)", fontsize=14, labelpad=11)

    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])
    if tick_interval is not None:
        ax.xaxis.set_major_locator(MultipleLocator(tick_interval))
        ax.yaxis.set_major_locator(MultipleLocator(tick_interval))
        ax.zaxis.set_major_locator(MultipleLocator(tick_interval))
    else:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.zaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(axis='x', pad=1, labelsize=7, labelrotation=20)
    ax.tick_params(axis='y',pad=3, labelsize=7)
    ax.tick_params(axis='z', pad=1, labelsize=7, labelrotation=20)
    ax.autoscale(False)      # 자동 스케일 끔

    plt.title(f"{motion_id}_{joint_name}_{save_name}")
    plt.colorbar(sc, pad=0.15)
    plt.savefig(
        generate_plot_path("Results/dribble", motion_id, joint_idx, joint_name, save_name, "1"),
        dpi=300,
    )
    plt.clf()
    plt.close(fig)
    
    # 2D subplots - 각 서브플롯에 올바른 축 범위 적용
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    # XY
    axes[0].plot(x, y, "+", color="r", markersize=3)
    axes[0].set_xlabel("X (flx, ext)", fontsize=13, fontweight='bold')
    axes[0].set_ylabel("Y (exr, inr)" , fontsize=13, fontweight='bold')
    axes[0].set_xlim(x_min, x_max)
    axes[0].set_ylim(y_min, y_max)
    if tick_interval is not None:
        axes[0].xaxis.set_major_locator(MultipleLocator(tick_interval))
        axes[0].yaxis.set_major_locator(MultipleLocator(tick_interval))
    else:
        axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
        axes[0].yaxis.set_major_locator(MaxNLocator(integer=True))
    for lbl in axes[0].get_xticklabels():
        lbl.set_fontsize(9)
        lbl.set_rotation(35)
    for lbl in axes[0].get_yticklabels():
        lbl.set_fontsize(9)
        lbl.set_rotation(35)
    axes[0].set_title("XY Plane")
    axes[0].grid(True, alpha=0.3)

    axes[0].scatter(
        x[0], y[0], s=70, marker='o', linewidths=1
        )
    axes[0].scatter(
        x[end_idx], y[end_idx], s=70, marker='X', linewidths=1
        )

    # XZ
    axes[1].plot(x, z, "+", color="g", markersize=3)
    axes[1].set_xlabel("X (flx, ext)", fontsize=13, fontweight='bold')
    axes[1].set_ylabel("Z (abb, abd)", fontsize=13, fontweight='bold')
    axes[1].set_xlim(x_min, x_max)
    axes[1].set_ylim(z_min, z_max)
    if tick_interval is not None:
        axes[1].xaxis.set_major_locator(MultipleLocator(tick_interval))
        axes[1].yaxis.set_major_locator(MultipleLocator(tick_interval))
    else:
        axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
        axes[1].yaxis.set_major_locator(MaxNLocator(integer=True))
    for lbl in axes[1].get_xticklabels():
        lbl.set_fontsize(9)
        lbl.set_rotation(35)
    for lbl in axes[1].get_yticklabels():
        lbl.set_fontsize(9)
        lbl.set_rotation(35)
    axes[1].set_title("XZ Plane")
    axes[1].grid(True, alpha=0.3)

    axes[1].scatter(
        x[0], z[0], s=70, marker='o', linewidths=1
        )
    axes[1].scatter(
        x[-1], z[-1], s=70, marker='X', linewidths=1
        )

    # YZ
    axes[2].plot(y, z, "+", color="b", markersize=3)
    axes[2].set_xlabel("Y (exr, inr)", fontsize=13, fontweight='bold')
    axes[2].set_ylabel("Z (abb, abd)", fontsize=13, fontweight='bold')
    axes[2].set_xlim(y_min, y_max)
    axes[2].set_ylim(z_min, z_max)
    if tick_interval is not None:
        axes[2].xaxis.set_major_locator(MultipleLocator(tick_interval))
        axes[2].yaxis.set_major_locator(MultipleLocator(tick_interval))
    else:
        axes[2].xaxis.set_major_locator(MaxNLocator(integer=True))
        axes[2].yaxis.set_major_locator(MaxNLocator(integer=True))
    for lbl in axes[2].get_xticklabels():
        lbl.set_fontsize(9)
        lbl.set_rotation(35)
    for lbl in axes[2].get_yticklabels():
        lbl.set_fontsize(9)
        lbl.set_rotation(35)
    axes[2].set_title("YZ Plane")
    axes[2].grid(True, alpha=0.3)

    axes[2].scatter(
        y[0], z[0], s=70, marker='o', linewidths=1
        )
    axes[2].scatter(
        y[-1], z[-1], s=70, marker='X', linewidths=1
        )

    plt.suptitle(f"{joint_idx+1}. {joint_name}", fontsize=12)
    plt.tight_layout()
    plt.savefig(
        generate_plot_path("Results/dribble", motion_id, joint_idx, joint_name, save_name, "2"),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()


def plot_joint_angles_per_frame(
    x, y, z, joint_idx, joint_name, motion_id,
    save_name="angles_per_frame", base_path="Results/dribble",
    axis_labels=None, y_axis_range=None
):
    if axis_labels is None:
        axis_labels = [
            'X (flx, ext)', 'Y (exr, inr)', 'Z (abb, abd)'
        ]

    # 리스트를 numpy 배열로 변환
    x_vals = np.array(x)
    y_vals = np.array(y)
    z_vals = np.array(z)

    num_frames = len(x_vals)
    frames = np.arange(num_frames)

    # Y축 범위 설정: global_axis_min/max의 X축 범위 사용
    axis_min, axis_max = y_axis_range
    ylim_min = float(axis_min[2])  # X축 min
    ylim_max = float(axis_max[2])  # X축 max


    # 3개의 서브플롯 생성 (Y, Z, X 각각)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # X축 각도
    axes[0].plot(frames, x_vals, 'r-', linewidth=1.5, label=axis_labels[0])
    axes[0].set_xlabel('Frame', fontsize=12, fontweight='bold')
    axes[0].set_ylabel(
        f'{axis_labels[0]}', fontsize=12, fontweight='bold'
    )
    # axes[0].set_title(f'{axis_labels[0]}', fontsize=11)
    axes[0].set_ylim(ylim_min, ylim_max)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Y축 각도
    axes[1].plot(frames, y_vals, 'g-', linewidth=1.5, label=axis_labels[1])
    axes[1].set_xlabel('Frame', fontsize=12, fontweight='bold')
    axes[1].set_ylabel(
        f'{axis_labels[1]}', fontsize=12, fontweight='bold'
    )
    # axes[1].set_title(f'{axis_labels[1]}', fontsize=11)
    axes[1].set_ylim(ylim_min, ylim_max)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Z축 각도
    axes[2].plot(frames, z_vals, 'b-', linewidth=1.5, label=axis_labels[2])
    axes[2].set_xlabel('Frame', fontsize=12, fontweight='bold')
    axes[2].set_ylabel(
        f'{axis_labels[2]}', fontsize=12, fontweight='bold'
    )
    # axes[2].set_title(f'{axis_labels[2]}', fontsize=11)
    axes[2].set_ylim(ylim_min, ylim_max)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.suptitle(
        f"{joint_idx+1}. {joint_name}",
        fontsize=14, fontweight='bold'
    )

    plt.subplots_adjust(hspace=0.6)  # 세로 간격 조절 (값이 클수록 간격이 넓어짐)
    # plt.tight_layout()

    # 저장 경로 생성 및 저장
    save_path = generate_plot_path(
        base_path, motion_id, joint_idx, joint_name, save_name, "angles"
    )
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


    # 모든 축을 하나의 그래프에 그리기
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.plot(
        frames, x_vals, 'r-', linewidth=1.5, label=axis_labels[0], alpha=0.8
    )
    ax.plot(
        frames, y_vals, 'g-', linewidth=1.5, label=axis_labels[1], alpha=0.8
    )
    ax.plot(
        frames, z_vals, 'b-', linewidth=1.5, label=axis_labels[2], alpha=0.8
    )
    ax.set_xlabel('Frame', fontsize=12, fontweight='bold')
    ax.set_ylabel('Angle (degrees)', fontsize=12, fontweight='bold')
    ax.set_title(
        f"{joint_idx+1}. {joint_name} - All Axis",
        fontsize=13, fontweight='bold'
    )
    ax.set_ylim(ylim_min, ylim_max)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    plt.tight_layout()

    # 저장 경로 생성 및 저장 (통합 그래프)
    save_path_combined = generate_plot_path(
        base_path, motion_id, joint_idx, joint_name, save_name,
        "angles_combined"
    )
    plt.savefig(save_path_combined, dpi=300, bbox_inches='tight')
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
