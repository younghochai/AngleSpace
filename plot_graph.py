import os
import json
import glob
from pathlib import Path
import numpy as np
from os.path import join as ospj
from tqdm import tqdm
from utils import (
    check_label_in_annotations,
    convert_pose_to_euler,
    lower_body_joints,
    global_joint_min,
    global_joint_max,
)
from utils_graph import (
    plot_joint_trajectory,
    plot_joint_angles_per_frame,
    plot_selected_right_joint_axes
)

global_axis_min = np.array([np.inf, np.inf, np.inf], dtype=float)  # [Y, Z, X]
global_axis_max = np.array([-np.inf, -np.inf, -np.inf], dtype=float)  # [Y, Z, X]


def ROM_from_BABEL():
    global global_axis_min, global_axis_max
    # 경로 설정
    d_folder = "./babel_v1.0_release"
    l_babel_dense_files = ["train"]

    babel = {}
    for file in l_babel_dense_files:
        babel[file] = json.load(open(ospj(d_folder, file + ".json")))

    # BABEL에서 ROM 찾기
    keylist = list(babel[file].keys())
    motion_list = []

    find_labels = ["leg movements", "knee movement", "sports move", "play sports", "exercise/training"]

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
        for label in find_labels:
            if check_label_in_annotations(data_entry, label):
                motion_list.append(keylist[kl])

    motion_list = set(motion_list)
    # print(len(motion_list))

    pathprefix = "/home/wonjinmon/문서/AMASS/smplx"

    missing_count = 0
    processed = 0

    for motion in tqdm(motion_list):
        data = babel[file].get(motion, {})
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
        # print(poses_axis.shape)

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

            axis_min_j = euler_all[:, j, :].min(axis=0)  # (3,) -> 해당 관절의 Y,Z,X 최소
            axis_max_j = euler_all[:, j, :].max(axis=0)  # (3,) -> 해당 관절의 Y,Z,X 최대
            global_axis_min = np.minimum(global_axis_min, axis_min_j)
            global_axis_max = np.maximum(global_axis_max, axis_max_j)

        processed += 1

    # 최종 min, max 출력
    print("\n" + "=" * 60)
    print("최종 Global Min/Max 값:")

    for j, jname in lower_body_joints.items():
        gmin = global_joint_min[j]
        gmax = global_joint_max[j]
        gr = gmax - gmin
        print(
            f"{jname:7s} :   "
            f"Y-[{gmin[0]:7.2f},{gmax[0]:7.2f}], Yrange: {gr[0]:6.2f}    "
            f"Z-[{gmin[1]:7.2f},{gmax[1]:7.2f}], Zrange: {gr[1]:6.2f}    "
            f"X-[{gmin[2]:7.2f},{gmax[2]:7.2f}], Xrange: {gr[2]:6.2f}"
        )
    print("\n" + "=" * 60)
    print("축별 Global Min/Max 및 Range (Y, Z, X):")
    # agr = global_axis_max - global_axis_min
    print(
        f"Y축: [{global_axis_min[0]:7.2f}, {global_axis_max[0]:7.2f}]  \n"
        f"Z축: [{global_axis_min[1]:7.2f}, {global_axis_max[1]:7.2f}]  \n"
        f"X축: [{global_axis_min[2]:7.2f}, {global_axis_max[2]:7.2f}]  "
    )

    return global_axis_min, global_axis_max


def plot_graph(path, min_val, max_val, tick_interval=None):
    npz_path = sorted(glob.glob(path))
    print(len(npz_path))

    for npz in tqdm(npz_path):
        motion_id = Path(npz).stem

        poses = np.load(npz)['poses'] # 156개
        print(poses.shape)
        poses = poses[..., :66] # body 63개만 사용

        # pose_body: (N, 63) -> (N, 21, 3)
        poses_axis = poses.reshape(-1, 22, 3)
        print(motion_id, poses_axis.shape)

        # (..,3) >>> YZX로
        euler_all = convert_pose_to_euler(poses_axis.reshape(-1, 3)).reshape(-1, 22, 3)

        # 그래프 그리기
        for j, jname in lower_body_joints.items():

            x = euler_all[:, j, 2].tolist()  # X
            y = euler_all[:, j, 0].tolist()  # Y
            z = euler_all[:, j, 1].tolist()  # Z

            plot_joint_trajectory(
                x, y, z, j, jname, 'sequence_v2', min_val, max_val,
                motion_id=motion_id, tick_interval=tick_interval
            )

            # plot_joint_angles_per_frame(
            #     x, y, z, j, jname, motion_id, 'graph',
            #     y_axis_range=(min_val, max_val)
            # )

            if jname in ["R_Hip", "R_Knee", "R_Ankle"]:
                plot_selected_right_joint_axes(
                    x, y, z, j, jname, motion_id,
                    base_path="Results/",
                    y_axis_range=(min_val, max_val)
                )


if __name__ == "__main__":
    path = "./Data/npz/*.npz"
    min_value, max_value = ROM_from_BABEL()
    plot_graph(path, min_value, max_value, tick_interval=15)

    # path = "./Data/dribble_analysis/npz/*.npz"
    # min_value, max_value = ROM_from_BABEL()
    # plot_graph(path, min_value, max_value, tick_interval=30)
