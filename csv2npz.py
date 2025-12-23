import numpy as np
import pandas as pd
import os
import re
import glob
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

input_csv_folder = '/home/wonjinmon/문서/AngleSpace/Data/csv'  # CSV 파일이 있는 폴더
template_npz     = '/home/wonjinmon/문서/AngleSpace/Data/Run C24 - quick side step left_poses.npz'
output_npz_folder = '/home/wonjinmon/문서/AngleSpace/Data/npz'  # 결과 NPZ를 저장할 폴더

# --- 1) 출력 폴더가 없으면 생성
os.makedirs(output_npz_folder, exist_ok=True)

# --- 2) SMPL 전체 관절 수 정의 ---
# Run C24 포맷의 총 관절 수를 확인하세요 (예: 52)
NUM_JOINTS = 52
POSE_DIMS  = NUM_JOINTS * 3  # 156 dims

# --- 3) 템플릿 NPZ 로드: 메타데이터 복사 ---
template = np.load(template_npz, allow_pickle=True)
gender = template['gender']
mocap_framerate = template['mocap_framerate']
betas = template['betas']          # (16,)
dmpl_dim = template['dmpls'].shape[1]  # e.g., 8

# --- 4) 모든 CSV 파일 찾기 ---
csv_files = glob.glob(os.path.join(input_csv_folder, '*.csv'))
print(f"Found {len(csv_files)} CSV files to convert")

# --- 5) 각 CSV 파일에 대해 변환 수행 ---
for csv_path in tqdm(csv_files, desc="Converting CSV to NPZ"):
    # 파일명만 추출해서 출력 NPZ 경로 생성
    filename = os.path.basename(csv_path)
    output_npz = os.path.join(output_npz_folder, os.path.splitext(filename)[0] + '.npz')
    print(output_npz)

    try:
        # --- CSV 로드 및 프레임 수 ---
        df = pd.read_csv(csv_path)
        T_csv = len(df)

        # --- pelvis 위치 (trans) 추출 ---
        trans = df[['pelvis_px','pelvis_py','pelvis_pz']].to_numpy()  # (T_csv, 3)
        
        # --- CSV로부터 poses 추출 (axis–angle) ---
        # 컬럼 형식 '<joint>_wx/wy/wz' 자동 감지
        axes_cols = [c for c in df.columns if re.search(r'_(wx|wy|wz)$', c)]
        joints_csv = []
        for c in axes_cols:
            jn = re.sub(r'_(wx|wy|wz)$', '', c)
            if jn not in joints_csv:
                joints_csv.append(jn)
        num_csv = len(joints_csv)  # 기대: 22
        
        # CSV poses 배열 생성 (T_csv, num_csv*3)
        poses_csv = np.zeros((T_csv, num_csv*3), dtype=np.float64)
        for i, jn in enumerate(joints_csv):
            poses_csv[:, i*3:(i+1)*3] = df[[f'{jn}_wx', f'{jn}_wy', f'{jn}_wz']].to_numpy()
        
        # --- 전체 poses 초기화 및 채우기 ---
        # CSV에 있는 22개 관절은 앞부분부터 순서대로, 나머지는 0으로 유지
        poses_full = np.zeros((T_csv, POSE_DIMS), dtype=np.float64)
        poses_full[:, :num_csv*3] = poses_csv
        
        # --- dmpls 기본값 생성 ---
        dmpls = np.zeros((T_csv, dmpl_dim), dtype=np.float64)
        
        # --- NPZ로 저장: Run C24와 동일한 키·순서 ---
        np.savez(
            output_npz,
            trans=trans,
            gender=gender,
            mocap_framerate=mocap_framerate,
            betas=betas,
            dmpls=dmpls,
            poses=poses_full,
        )
        
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")

print(f"Conversion complete! Saved {len(csv_files)} files to {output_npz_folder}")