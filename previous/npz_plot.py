import glob
import numpy as np
from pathlib import Path
from utils import *
from tqdm import tqdm


npz_path = sorted(glob.glob("./Data/npz/*.npz"))

# 이 파일들은 156차원, Pose_body가 아니라 pose에 저장되어있음.
for npz in tqdm(npz_path):  # 하체 데이터만
    motion_id = Path(npz).stem 
    # print(motion_id)

    poses = np.load(npz)['poses'] # 156개
    poses = poses[..., :66] # body 63개만 사용

    # pose_body: (N, 63) -> (N, 21, 3)
    poses_axis = poses.reshape(-1, 22, 3)
    
    # (..,3) >>> YZX로
    euler_all = convert_pose_to_euler(poses_axis.reshape(-1, 3)).reshape(-1, 22, 3)

    # 하체 관절만 글로벌 min/max 업데이트
    for j, jname in lower_body_joints.items():
        j_min = euler_all[:, j, :].min(axis=0)
        j_max = euler_all[:, j, :].max(axis=0)
        global_joint_min[j] = np.minimum(global_joint_min[j], j_min)
        global_joint_max[j] = np.maximum(global_joint_max[j], j_max)
        gmin = global_joint_min[j]
        gmax = global_joint_max[j]
        gr  = gmax - gmin

# 최종 min, max 출력
print("\n" + "=" * 60)
print("최종 Global Min/Max 값:")
print("=" * 60)

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

for npz in tqdm(npz_path):
    motion_id = Path(npz).stem 

    poses = np.load(npz)['poses'] # 156개
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

        plot_joint_trajectory(x, y, z, j, jname, 'sequence', motion_id=motion_id)
