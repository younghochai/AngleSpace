import os
import numpy as np
import pandas as pd

# npz_path = 'C:/Users/Velab/Desktop/250922_Primitives/npz/a0046.npz'    # 입력 AMASS-style NPZ 파일 경로
# npz_path = "/home/wonjinmon/Save_CSV/AxisAngleData_251026_squat1.csv"
# output_csv = 'C:/Users/Velab/Desktop/250922_Primitives/a0046_test.csv' # 출력 CSV 파일 경로

# 1) NPZ 로드
data = np.load(npz_path, allow_pickle=True)
if "poses" not in data or "trans" not in data:
    raise KeyError("Input NPZ must contain 'poses' and 'trans' arrays.")
poses = data["poses"]  # (T, N) axis-angle
trans = data["trans"]  # (T, 3) root translation
T = poses.shape[0]

# 1.1) 모션 캡처 프레임 속도 로드 (Time 계산용)
# SMPL, SMPL-H, SMPL-X 등 다양한 키 이름 지원
frame_rate = data.get("mocap_frame_rate", None)
if frame_rate is None:
    frame_rate = data.get("mocap_framerate", None)
if frame_rate is None:
    raise KeyError(
        "Input NPZ must contain 'mocap_frame_rate' or 'mocap_framerate' array to compute Time column."
    )

# 2) 축-각(axis-angle) 재구성: (T, num_joints, 3)(axis-angle) 재구성: (T, num_joints, 3)
num_joints = poses.shape[1] // 3
if poses.shape[1] % 3 != 0:
    raise ValueError(
        f"Invalid poses shape: {poses.shape}. Second dimension must be divisible by 3."
    )
axis_full = poses.reshape(T, num_joints, 3)

# 3) 포지션 재구성: (T, num_joints, 3)
pos_full = np.zeros((T, num_joints, 3))
pos_full[:, 0, :] = trans

# 4) 원래 22개 joint만 추출
if num_joints < 22:
    raise ValueError(f"Not enough joints: found {num_joints}, need at least 22.")
axis22 = axis_full[:, :22, :]  # (T, 22, 3)
pos22 = pos_full[:, :22, :]

# 5) angles와 positions 별도 평탄화
angle_flat = axis22.reshape(T, 22 * 3)
pos_flat = pos22.reshape(T, 22 * 3)
flat = np.hstack([angle_flat, pos_flat])

# 6) 프레임 및 시간 컬럼 생성
frames = np.arange(T)
times = frames / frame_rate

# 7) 사용자 지정 관절 순서
joints_order = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]


# 8) 컬럼 이름 생성
def make_labels(names, suffixes):
    return [f"{name}_{suf}" for name in names for suf in suffixes]


angle_labels = make_labels(joints_order, ["wx", "wy", "wz"])
pos_labels = make_labels(joints_order, ["px", "py", "pz"])
columns = ["Frame", "Time"] + angle_labels + pos_labels

# 9) DataFrame 생성 및 CSV 저장
df = pd.DataFrame(flat, columns=angle_labels + pos_labels)
df.insert(0, "Time", times)
df.insert(0, "Frame", frames)
df = df[columns]
df.to_csv(output_csv, index=False)
print(f"Saved joints CSV : {os.path.abspath(output_csv)}")
