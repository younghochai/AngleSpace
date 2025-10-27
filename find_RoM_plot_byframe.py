# frame_anns 단위로 plot하기

# 관절 움직임의 최댓값과 최솟값 확인하기 (프레임 단위 처리)
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from os.path import join as ospj
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

from utils import *


def plot_joint_trajectory(x, y, z, joint_idx, joint_name, save_name, motion_id):
    """프레임별 처리 후 플롯 생성"""
    # 글로벌 min/max를 기반으로 축 범위 설정
    gmin = global_min[joint_idx]  # [Y,Z,X]
    gmax = global_max[joint_idx]  # [Y,Z,X]
    x_min, x_max = float(gmin[2]), float(gmax[2])  # X
    y_min, y_max = float(gmin[0]), float(gmax[0])  # Y
    z_min, z_max = float(gmin[1]), float(gmax[1])  # Z

    # min==max 같은 퇴화 케이스 대비 소폭 패딩
    def _pad(lo, hi, eps=1e-6, frac=0.05):
        if hi - lo < eps:
            m = (hi + lo) / 2.0
            r = max(abs(m) * frac, 1.0)  # 최소 1도
            return m - r, m + r
        return lo, hi

    x_min, x_max = _pad(x_min, x_max)
    y_min, y_max = _pad(y_min, y_max)
    z_min, z_max = _pad(z_min, z_max)

    # 3D 플롯
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

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])
    plt.title(f"{joint_idx+1}{joint_name}__{save_name}")
    plt.colorbar(sc)
    # plt.show()
    plt.savefig(
        generate_plot_path("Output", motion_id, joint_idx, joint_name, save_name, "1"),
        dpi=300,
    )
    plt.clf()
    plt.close(fig)

    # 2D subplots
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    # XY 평면
    axes[0].plot(x, y, "+", color="r", markersize=3)
    axes[0].set_xlabel("X (Roll)")
    axes[0].set_ylabel("Y (Yaw)")
    axes[0].set_xlim(x_min, x_max)
    axes[0].set_ylim(y_min, y_max)
    axes[0].set_title("XY Plane")
    axes[0].grid(True, alpha=0.3)

    # XZ 평면
    axes[1].plot(x, z, "+", color="g", markersize=3)
    axes[1].set_xlabel("X (Roll)")
    axes[1].set_ylabel("Z (Pitch)")
    axes[1].set_xlim(x_min, x_max)
    axes[1].set_ylim(z_min, z_max)
    axes[1].set_title("XZ Plane")
    axes[1].grid(True, alpha=0.3)

    # YZ 평면
    axes[2].plot(y, z, "+", color="b", markersize=3)
    axes[2].set_xlabel("Y (Yaw)")
    axes[2].set_ylabel("Z (Pitch)")
    axes[2].set_xlim(y_min, y_max)
    axes[2].set_ylim(z_min, z_max)
    axes[2].set_title("YZ Plane")
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(f"{joint_idx+1}. {joint_name} - 2D Projections", fontsize=12)
    plt.tight_layout()
    plt.savefig(
        generate_plot_path("Output", motion_id, joint_idx, joint_name, save_name, "2"),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()


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
global_min = {
    j: np.array([np.inf, np.inf, np.inf], dtype=float)
    for j in lower_body_joints.keys()
}
global_max = {
    j: np.array([-np.inf, -np.inf, -np.inf], dtype=float)
    for j in lower_body_joints.keys()
}

if __name__ == "__main__":

    # 경로 설정
    d_folder = "./babel_v1.0_release"
    l_babel_dense_files = ["extra_val"]
    # l_babel_extra_files = ["extra_train"]

    babel = {}
    for file in l_babel_dense_files:
        babel[file] = json.load(open(ospj(d_folder, file + ".json")))

    keylist = list(babel[file].keys())

    leg_list = []

    findLabel_1 = "leg movements"
    frameCount = 0

    for kl in range(len(keylist)):
        inFrameAnn = babel[file][keylist[kl]]["frame_ann"]

        if inFrameAnn is not None:
            frameCount += 1
            labels_list = babel[file][keylist[kl]]["frame_ann"]["labels"]

            FindLabel_1 = False

            for ll in range(len(labels_list)):
                act_cat_list = babel[file][keylist[kl]]["frame_ann"]["labels"][ll]["act_cat"]

                for al in range(len(act_cat_list)):
                    tempstr = babel[file][keylist[kl]]["frame_ann"]["labels"][ll]["act_cat"][al]

                    if findLabel_1 in tempstr:
                        FindLabel_1 = True

            if FindLabel_1:
                leg_list.append(keylist[kl])

    pathprefix = "/home/wonjinmon/문서/AMASS/smplx"

    missing_count = 0
    processed = 0

    # leg_list에서 처리할 범위 설정
    for leg_id in tqdm(leg_list):
        data = babel[file].get(leg_id, {})
        feat_p = str(data.get("feat_p", ""))
        if not feat_p:
            continue

        # 프레임 어노테이션 정보 가져오기
        ann = data.get('frame_ann', {})
        if not ann or 'labels' not in ann:
            continue

        # 경로 수정
        parts = feat_p.split("/")
        fname = parts[-1].replace("poses", "stageii")
        path = os.path.join(pathprefix, *parts[1:-1], fname)

        # 예외처리
        if not os.path.isfile(path) and "BioMotionLab_NTroje" in parts:
            parts = [t for t in parts if t != "BioMotionLab_NTroje"]
            path = os.path.join(pathprefix + '/BMLrub', *parts[1:-1], fname)
        if not os.path.isfile(path) and "/MPI_Limits" in path:
            path = path.replace("/MPI_Limits", "/PosePrior")
        if not os.path.isfile(path) and "/MPI_mosh" in path:
            path = path.replace("/MPI_mosh", "/MoSh")
        if not os.path.isfile(path) and ' ' in path:
            path = path.replace(' ', '_')
        if not os.path.isfile(path):
            missing_count += 1
            print(f"File not found: {path}")
            continue

        # pose_body 로드
        np_all = np.load(path)["pose_body"]
        print(f"Loaded pose data shape: {np_all.shape}")

        # 각 레이블별로 처리 (frame_ann 단위)
        for label in ann['labels']:
            act_cat = label.get('act_cat', [])

            # leg movements 확인
            if not any('leg movements' in cat for cat in act_cat):
                continue

            # 프레임 범위 계산 (120 FPS 가정)
            start_frame = int(label['start_t'] * 120)
            end_frame = int(label['end_t'] * 120)
            save_plot_name = f"{label['start_t']}_{label['end_t']}"
            print(f"Processing frames {start_frame} to {end_frame}")

            # 각 관절별로 프레임 단위 처리
            for joint_idx in lower_body_joints.keys():
                x, y, z = [], [], []

                # 프레임별로 순회하며 오일러 각도 계산
                for i in range(start_frame, end_frame + 1):
                    if i >= len(np_all):
                        continue

                    # 해당 프레임의 특정 관절 pose 추출
                    pose = np_all[i].reshape(21, 3)[joint_idx]

                    # 오일러 각도 변환
                    euler = convert_pose_to_euler(pose)
                    x.append(euler[2])  # X축 (Roll)
                    y.append(euler[0])  # Y축 (Yaw)
                    z.append(euler[1])  # Z축 (Pitch)

                # 글로벌 min/max 업데이트
                if x and y and z:
                    x_arr = np.array(x)
                    y_arr = np.array(y)
                    z_arr = np.array(z)

                    # [Y, Z, X] 순서로 저장
                    local_min = np.array([y_arr.min(), z_arr.min(), x_arr.min()])
                    local_max = np.array([y_arr.max(), z_arr.max(), x_arr.max()])

                    global_min[joint_idx] = np.minimum(global_min[joint_idx], local_min)
                    global_max[joint_idx] = np.maximum(global_max[joint_idx], local_max)

                # 그래프 그리기
                joint_name = lower_body_joints[joint_idx]
                plot_joint_trajectory(x, y, z, joint_idx, joint_name, save_plot_name, leg_id)

            print(f"Completed: {save_plot_name}")

        for j, jname in lower_body_joints.items():
            gmin = global_min[j]
            gmax = global_max[j]
            print(
                f"{jname:12s} :      "
                f"Y-[{gmin[0]:7.2f},{gmax[0]:7.2f}]  "
                f"Z-[{gmin[1]:7.2f},{gmax[1]:7.2f}]  "
                f"X-[{gmin[2]:7.2f},{gmax[2]:7.2f}]"
            )

        processed += 1

    print(f"\nProcessed: {processed}, Missing files: {missing_count}")
