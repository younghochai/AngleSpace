import os
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator, MultipleLocator


right_leg_cache = {}


def generate_plot_path(
    base_path, motion_id, joint_idx, joint_name, save_plot_name, suffix
):
    folder_path = os.path.join(base_path, str(motion_id))
    os.makedirs(folder_path, exist_ok=True)
    return os.path.join(
        folder_path,
        f"{motion_id}_{joint_idx+1}{joint_name}_{save_plot_name}_{suffix}.png",
    )


def plot_joint_trajectory(
    x,
    y,
    z,
    joint_idx,
    joint_name,
    save_name,
    min_vals,
    max_vals,
    motion_id,
    tick_interval=None,
    frame_marks=None,
):
    # 3축 전체에서 공통 min / max 찾기
    raw_min = float(np.min(min_vals))
    raw_max = float(np.max(max_vals))

    global_min = math.floor(raw_min)  # 최소는 바닥으로
    global_max = math.ceil(raw_max)  # 최대는 천장으로

    x_min, x_max = int(global_min), int(global_max)
    y_min, y_max = int(global_min), int(global_max)
    z_min, z_max = int(global_min), int(global_max)

    if frame_marks is None:
        frame_marks = [0, 8, 16, 29, 43, 59]
    n = len(x)
    frame_marks = [f for f in frame_marks if 0 <= f < n]

    valid = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(z)
    end_idx = np.flatnonzero(valid)[-1] if np.any(valid) else -1

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(x, y, z, c=z, alpha=0.7, depthshade=True)

    for seg_idx, f in enumerate(frame_marks, start=1):
        ax.scatter(
            x[f],
            y[f],
            z[f],
            s=60,
            marker="D",  # 다이아몬드 마커
            edgecolors="k",
            facecolors="none",
            linewidths=1.2,
        )
        ax.text(x[f], y[f], z[f], f"S{seg_idx}", fontsize=6, fontweight="bold")

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

    ax.set_xlim([x_min + 30, x_max - 115])
    ax.set_ylim([y_min + 30, y_max - 115])
    ax.set_zlim([z_min + 30, z_max - 115])

    if tick_interval is not None:
        ax.xaxis.set_major_locator(MultipleLocator(tick_interval))
        ax.yaxis.set_major_locator(MultipleLocator(tick_interval))
        ax.zaxis.set_major_locator(MultipleLocator(tick_interval))
    else:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.zaxis.set_major_locator(MaxNLocator(integer=True))

    ax.tick_params(axis="x", pad=1, labelsize=7, labelrotation=20)
    ax.tick_params(axis="y", pad=3, labelsize=7)
    ax.tick_params(axis="z", pad=1, labelsize=7, labelrotation=20)
    ax.autoscale(False)  # 자동 스케일 끔

    plt.title(f"{motion_id}_{joint_name}_{save_name}")
    plt.colorbar(sc, pad=0.15)
    plt.savefig(
        generate_plot_path("Results", motion_id, joint_idx, joint_name, save_name, "1"),
        dpi=300,
    )
    plt.clf()
    plt.close(fig)

    # 2D subplots - 각 서브플롯에 올바른 축 범위 적용
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    # XY
    axes[0].plot(x, y, "+", color="r", markersize=3)
    axes[0].set_xlabel("X (flx, ext)", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Y (exr, inr)", fontsize=13, fontweight="bold")
    axes[0].set_xlim(x_min + 30, x_max - 115)
    axes[0].set_ylim(y_min + 30, y_max - 115)
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

    # axes[0].scatter(
    #     x[0], y[0], s=70, marker='o', linewidths=1
    #     )
    # axes[0].scatter(
    #     x[end_idx], y[end_idx], s=70, marker='X', linewidths=1
    #     )

    for idx, f in enumerate(frame_marks, start=1):
        axes[0].scatter(
            x[f],
            y[f],
            s=35,
            marker="D",
            edgecolors="k",
            facecolors="none",
            linewidths=1,
        )
        axes[0].text(
            x[f], y[f], f" {idx}", fontsize=9, fontweight="bold", ha="left", va="bottom"
        )

    # XZ
    axes[1].plot(x, z, "+", color="g", markersize=3)
    axes[1].set_xlabel("X (flx, ext)", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Z (abb, abd)", fontsize=13, fontweight="bold")
    axes[1].set_xlim(x_min + 30, x_max - 115)
    axes[1].set_ylim(z_min + 30, z_max - 115)
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

    # axes[1].scatter(
    #     x[0], z[0], s=70, marker='o', linewidths=1
    #     )
    # axes[1].scatter(
    #     x[-1], z[-1], s=70, marker='X', linewidths=1
    #     )

    for idx, f in enumerate(frame_marks, start=1):
        axes[1].scatter(
            x[f],
            z[f],
            s=35,
            marker="D",
            edgecolors="k",
            facecolors="none",
            linewidths=1,
        )
        axes[1].text(
            x[f], z[f], f" {idx}", fontsize=9, fontweight="bold", ha="left", va="bottom"
        )

    # YZ
    axes[2].plot(y, z, "+", color="b", markersize=3)
    axes[2].set_xlabel("Y (exr, inr)", fontsize=13, fontweight="bold")
    axes[2].set_ylabel("Z (abb, abd)", fontsize=13, fontweight="bold")
    axes[2].set_xlim(y_min + 30, y_max - 115)
    axes[2].set_ylim(z_min + 30, z_max - 115)
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

    # axes[2].scatter(
    #     y[0], z[0], s=70, marker='o', linewidths=1
    #     )
    # axes[2].scatter(
    #     y[-1], z[-1], s=70, marker='X', linewidths=1
    #     )

    for idx, f in enumerate(frame_marks, start=1):
        axes[2].scatter(
            y[f],
            z[f],
            s=35,
            marker="D",
            edgecolors="k",
            facecolors="none",
            linewidths=1,
        )
        axes[2].text(
            y[f], z[f], f" {idx}", fontsize=9, fontweight="bold", ha="left", va="bottom"
        )

    plt.suptitle(f"{joint_idx+1}. {joint_name}", fontsize=12)
    plt.tight_layout()
    plt.savefig(
        generate_plot_path("Results", motion_id, joint_idx, joint_name, save_name, "2"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_joint_angles_per_frame(
    x,
    y,
    z,
    joint_idx,
    joint_name,
    motion_id,
    save_name="angles_per_frame",
    base_path="Results/",
    axis_labels=None,
    y_axis_range=None,
):
    if axis_labels is None:
        axis_labels = ["X (flx, ext)", "Y (exr, inr)", "Z (abb, abd)"]

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
    axes[0].plot(frames, x_vals, "r-", linewidth=1.5, label=axis_labels[0])
    axes[0].set_xlabel("Frame", fontsize=12, fontweight="bold")
    axes[0].set_ylabel(f"{axis_labels[0]}", fontsize=12, fontweight="bold")
    # axes[0].set_title(f'{axis_labels[0]}', fontsize=11)
    axes[0].set_ylim(ylim_min, ylim_max)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Y축 각도
    axes[1].plot(frames, y_vals, "g-", linewidth=1.5, label=axis_labels[1])
    axes[1].set_xlabel("Frame", fontsize=12, fontweight="bold")
    axes[1].set_ylabel(f"{axis_labels[1]}", fontsize=12, fontweight="bold")
    # axes[1].set_title(f'{axis_labels[1]}', fontsize=11)
    axes[1].set_ylim(ylim_min, ylim_max)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Z축 각도
    axes[2].plot(frames, z_vals, "b-", linewidth=1.5, label=axis_labels[2])
    axes[2].set_xlabel("Frame", fontsize=12, fontweight="bold")
    axes[2].set_ylabel(f"{axis_labels[2]}", fontsize=12, fontweight="bold")
    # axes[2].set_title(f'{axis_labels[2]}', fontsize=11)
    axes[2].set_ylim(ylim_min, ylim_max)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.suptitle(f"{joint_idx+1}. {joint_name}", fontsize=14, fontweight="bold")

    plt.subplots_adjust(hspace=0.6)  # 세로 간격 조절 (값이 클수록 간격이 넓어짐)
    # plt.tight_layout()

    # 저장 경로 생성 및 저장
    save_path = generate_plot_path(
        base_path, motion_id, joint_idx, joint_name, save_name, "angles"
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 모든 축을 하나의 그래프에 그리기
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.plot(frames, x_vals, "r-", linewidth=1.5, label=axis_labels[0], alpha=0.8)
    ax.plot(frames, y_vals, "g-", linewidth=1.5, label=axis_labels[1], alpha=0.8)
    ax.plot(frames, z_vals, "b-", linewidth=1.5, label=axis_labels[2], alpha=0.8)
    ax.set_xlabel("Frame", fontsize=12, fontweight="bold")
    ax.set_ylabel("Angle (degrees)", fontsize=12, fontweight="bold")
    ax.set_title(
        f"{joint_idx+1}. {joint_name} - All Axis", fontsize=13, fontweight="bold"
    )
    ax.set_ylim(ylim_min, ylim_max)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()

    # 저장 경로 생성 및 저장 (통합 그래프)
    save_path_combined = generate_plot_path(
        base_path, motion_id, joint_idx, joint_name, save_name, "angles_combined"
    )
    plt.savefig(save_path_combined, dpi=300, bbox_inches="tight")
    plt.close()


def plot_selected_right_joint_axes(
    x,
    y,
    z,
    joint_idx,
    joint_name,
    motion_id,
    base_path="Results/",
    y_axis_range=None,
    axis_labels=None,
):
    """
    오른쪽 힙: X, Z
    오른쪽 무릎: X
    오른쪽 발목: X

    - 각 관절/축별로 개별 그래프 저장
    - 추가로 R_Hip X/Z + R_Knee X + R_Ankle X 를
      한 그래프에 동시에 그려서 한 장 더 저장
    """
    global right_leg_cache

    if axis_labels is None:
        axis_labels = ["X (flx, ext)", "Y (exr, inr)", "Z (abb, abd)"]

    x_vals = np.array(x)
    y_vals = np.array(y)
    z_vals = np.array(z)
    frames = np.arange(len(x_vals))

    # ----- Y축 범위: 무조건 X축 범위를 기준으로 사용 -----
    if y_axis_range is not None:
        axis_min, axis_max = y_axis_range  # (Y,Z,X)
        ylim_min = float(axis_min[2])  # X축 min
        ylim_max = float(axis_max[2])  # X축 max
    else:
        # global 안 들어온 경우에는 이 joint의 X값 기준
        ylim_min = float(np.min(x_vals))
        ylim_max = float(np.max(x_vals))
    # -------------------------------------------------------

    def _add_segment_lines(ax):
        """수직 점선 + segment 번호 1~6"""
        segment_frames = [0, 9, 17, 30, 43, 51]
        for seg_idx, f in enumerate(segment_frames, start=1):
            if 0 <= f < len(frames):
                ax.axvline(x=f, linestyle=":", linewidth=1.0)
                y_top = ax.get_ylim()[1]
                ax.text(
                    f,
                    y_top,
                    str(seg_idx),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

    def _plot_single(frames, values, label, color, save_plot_name):
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        ax.plot(frames, values, color + "-", linewidth=1.5, label=label)
        ax.set_xlabel("Frame", fontsize=12, fontweight="bold")
        ax.set_ylabel("Angle (degrees)", fontsize=12, fontweight="bold")
        ax.set_title(
            f"                {joint_idx+1}. {joint_name} - {label}",
            fontsize=13,
            fontweight="bold",
        )

        # ax.set_ylim(ylim_min, ylim_max)
        ax.set_ylim(-90, 60)
        ax.yaxis.set_major_locator(MultipleLocator(15))

        _add_segment_lines(ax)

        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        plt.tight_layout()

        save_path = generate_plot_path(
            base_path, motion_id, joint_idx, joint_name, save_plot_name, "axis"
        )
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    # ====== 1) 관절별 개별 그래프 ======
    if joint_name == "R_Hip":
        # X
        _plot_single(frames, x_vals, axis_labels[0], "r", save_plot_name="R_Hip_X")
        # Z
        _plot_single(frames, z_vals, axis_labels[2], "b", save_plot_name="R_Hip_Z")

    elif joint_name == "R_Knee":
        _plot_single(frames, x_vals, axis_labels[0], "y", save_plot_name="R_Knee_X")

    elif joint_name == "R_Ankle":
        _plot_single(frames, x_vals, axis_labels[0], "g", save_plot_name="R_Ankle_X")

    # ====== 2) 나중에 한 장짜리 그래프 그리기 위해 캐시에 저장 ======
    if joint_name in ("R_Hip", "R_Knee", "R_Ankle"):
        right_leg_cache[joint_name] = {
            "frames": frames,
            "x": x_vals,
            "z": z_vals,
        }

    # 세 개(R_Hip, R_Knee, R_Ankle) 다 들어왔으면 통합 그래프 한 장 그림
    if all(k in right_leg_cache for k in ("R_Hip", "R_Knee", "R_Ankle")):
        hip = right_leg_cache["R_Hip"]
        knee = right_leg_cache["R_Knee"]
        ankle = right_leg_cache["R_Ankle"]

        frames_c = hip["frames"]  # 길이 같다고 가정

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        # R_Hip X (빨강)
        ax.plot(frames_c, hip["x"], "r-", linewidth=1.5, label="R_Hip X")
        # R_Hip Z (파랑)
        ax.plot(frames_c, hip["z"], "b-", linewidth=1.5, label="R_Hip Z")
        # R_Knee X (노랑)
        ax.plot(frames_c, knee["x"], "y-", linewidth=1.5, label="R_Knee X")
        # R_Ankle X (초록)
        ax.plot(frames_c, ankle["x"], "g-", linewidth=1.5, label="R_Ankle X")

        ax.set_xlabel("Frame", fontsize=12, fontweight="bold")
        ax.set_ylabel("Angle (degrees)", fontsize=12, fontweight="bold")
        # ax.set_title(
        #     "Right leg joint angles (R_Hip X/Z, R_Knee X, R_Ankle X)",
        #     fontsize=13, fontweight='bold'
        # )
        ax.set_xlim(-5, 125)
        # ax.set_ylim(ylim_min, ylim_max)
        ax.set_ylim(-90, 60)
        ax.yaxis.set_major_locator(MultipleLocator(15))

        _add_segment_lines(ax)

        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        plt.tight_layout()

        # 저장 경로 (통합 그래프용 별도 이름)
        folder_path = os.path.join(base_path, str(motion_id))
        os.makedirs(folder_path, exist_ok=True)
        save_path_combined = os.path.join(
            folder_path, f"{motion_id}_RightLeg_selected_axes_combined_axis.png"
        )
        plt.savefig(save_path_combined, dpi=300, bbox_inches="tight")
        plt.close()

        # 다음 motion_id 위해 캐시 비우기
        right_leg_cache = {}
