# babel annotation을 활용하여 npz to AngleSpace 시각화
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from os.path import join as ospj
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from utils import *


def plot_joint_trajectory(x, y, z, joint_idx, joint_name, save_name):
    gmin = global_min[joint_idx]  # [Y,Z,X]
    gmax = global_max[joint_idx]  # [Y,Z,X]
    x_min, x_max = float(gmin[2]), float(gmax[2])  # X
    y_min, y_max = float(gmin[0]), float(gmax[0])  # Y
    z_min, z_max = float(gmin[1]), float(gmax[1])  # Z

    # min=max 같은 케이스 대비 패딩
    def _pad(lo, hi, eps=1e-6, frac=0.05):
        return lo, hi
    x_min, x_max = _pad(x_min, x_max)
    y_min, y_max = _pad(y_min, y_max)
    z_min, z_max = _pad(z_min, z_max)
    
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
        generate_plot_path("Output1021", leg_id, joint_idx, joint_name, save_name, "1"),
        dpi=300,
    )
    plt.clf()
    plt.close(fig)
    
    # 2D subplots - 각 서브플롯에 올바른 축 범위 적용
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
        generate_plot_path("Output1021", leg_id, joint_idx, joint_name, save_name, "2"),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()



if __name__ == "__main__":
    # 경로 설정
    d_folder = "./babel_v1.0_release"
    l_babel_dense_files = ["extra_val"]

    babel = {}
    for file in l_babel_dense_files:
        babel[file] = json.load(open(ospj(d_folder, file + ".json")))

    keylist = list(babel[file].keys())

    leg_list = []
    findLabel_1 = "leg movements"
    
    frame_ann_count = 0
    seq_ann_count = 0

    for kl in range(len(keylist)):
        data_entry = babel[file][keylist[kl]]
        
        # annotation 타입 카운트
        if data_entry.get("frame_ann") is not None:
            frame_ann_count += 1
        elif data_entry.get("seq_anns") is not None:
            seq_ann_count += 1
        
        # 라벨 체크
        if check_label_in_annotations(data_entry, findLabel_1):
            leg_list.append(keylist[kl])
    
    # print((leg_list))
    # print(leg_list[6:7])

    # import sys
    # sys.exit()

    pathprefix = "/home/wonjinmon/문서/AMASS/smplx"

    missing_count = 0
    processed = 0

    for leg_id in tqdm(leg_list[6:7]):
        data = babel[file].get(leg_id, {})
        feat_p = str(data.get("feat_p", ""))
        if not feat_p:
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

        # 마지막 확인
        if not os.path.isfile(path):
            missing_count += 1
            print(path)
            continue

        # pose_body: (N, 63) -> (N, 21, 3)
        poses_axis = np.load(path)["pose_body"].reshape(-1, 21, 3)
        print(poses_axis.shape)

        # (..,3) >>> YZX로
        euler_all = convert_pose_to_euler(poses_axis.reshape(-1, 3)).reshape(-1, 21, 3)

        # 하체 관절만 글로벌 min/max 업데이트
        for j, jname in lower_body_joints.items():
            j_min = euler_all[:, j, :].min(axis=0)
            j_max = euler_all[:, j, :].max(axis=0)
            global_joint_min[j] = np.minimum(global_joint_min[j], j_min)
            global_joint_max[j] = np.maximum(global_joint_max[j], j_max)
            gmin = global_joint_min[j]
            gmax = global_joint_max[j]
            print(
                f"{jname} :      "
                f"Y-[{gmin[0]:7.2f},{gmax[0]:7.2f}]  "
                f"Z-[{gmin[1]:7.2f},{gmax[1]:7.2f}]  "
                f"X-[{gmin[2]:7.2f},{gmax[2]:7.2f}]"
            )
        processed += 1

        # 그래프 그리기
        for j, jname in tqdm(lower_body_joints.items()):

            x = euler_all[:, j, 2].tolist()  # X
            y = euler_all[:, j, 0].tolist()  # Y
            z = euler_all[:, j, 1].tolist()  # Z

            # 저장 파일명만 구분되게
            plot_joint_trajectory(x, y, z, j, jname, 'sequence')