import os
import json
import glob
from pathlib import Path
import numpy as np
from os.path import join as ospj
from tqdm import tqdm
import matplotlib.pyplot as plt  # 추가

from utils import (
    convert_pose_to_euler,
    lower_body_joints,
    global_joint_min,
    global_joint_max,
)


def plot_graph(path):
    npz_path = sorted(glob.glob(path))
    print(len(npz_path))

    # 결과 저장 폴더
    save_root = "./Results/simple_xyz"
    os.makedirs(save_root, exist_ok=True)

    for npz in tqdm(npz_path):
        motion_id = Path(npz).stem

        poses = np.load(npz)["poses"]  # (N, 156) 같은 형식이라 가정
        poses = poses[..., :66]  # body 63개만 사용 (22*3)

        # pose_body: (N, 63) -> (N, 22, 3)
        poses_axis = poses.reshape(-1, 22, 3)
        print(poses.shape, motion_id, poses_axis.shape)

        # (..,3) >>> YZX로
        euler_all = convert_pose_to_euler(poses_axis.reshape(-1, 3)).reshape(
            -1, 22, 3
        )  # (T, 22, 3)

        T = euler_all.shape[0]
        frames = np.arange(T)  # 0 ~ T-1 프레임 인덱스

        # 이 motion 결과 저장 폴더
        motion_save_dir = ospj(save_root, motion_id)
        os.makedirs(motion_save_dir, exist_ok=True)

        # x축 눈금(프레임 숫자) 설정: 프레임 많으면 10개 정도만
        step = max(1, T // 10)
        xticks = np.arange(0, T, step)

        # 관절별 그래프 그리기
        for j, jname in lower_body_joints.items():
            # convert_pose_to_euler에서 반환한 축 순서가 [Y, Z, X]
            X = euler_all[:, j, 2]  # X
            Y = euler_all[:, j, 0]  # Y
            Z = euler_all[:, j, 1]  # Z

            fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
            # 서브플롯 사이 간격 줄이기
            plt.subplots_adjust(hspace=0.15)

            # 공통 y범위, x범위, x틱 설정
            for ax in axes:
                ax.set_ylim(-120, 170)  # 범위 고정
                ax.set_xlim(0, T - 1)
                ax.set_xticks(xticks)  # 프레임 숫자 찍기

            # X축 (빨강)
            axes[0].plot(frames, X, color="r")
            axes[0].set_ylabel("X angle (deg)")
            axes[0].set_title(f"{motion_id} - {jname} (joint {j})")

            # Y축 (초록)
            axes[1].plot(frames, Y, color="g")
            axes[1].set_ylabel("Y angle (deg)")

            # Z축 (파랑)
            axes[2].plot(frames, Z, color="b")
            axes[2].set_ylabel("Z angle (deg)")
            axes[2].set_xlabel("Frame")

            # 저장
            save_path = ospj(motion_save_dir, f"{motion_id}_j{j:02d}_{jname}_xyz.png")
            plt.tight_layout()
            plt.savefig(save_path, dpi=200)
            plt.close(fig)


if __name__ == "__main__":
    path = "./Data/npz/*.npz"
    plot_graph(path)
