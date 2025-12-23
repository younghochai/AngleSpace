# euler_to_aa_csv.py

import argparse
import math
import numpy as np
import pandas as pd


def find_joint_bases(columns, euler_suffixes):
    """
    컬럼 순서를 그대로 따라가면서
    '<base>_X/Y/Z' 세트를 가진 관절 base 이름들을 찾음.
    """
    s1, s2, s3 = euler_suffixes
    cols_set = set(columns)
    bases = []
    seen = set()

    for col in columns:
        if not col.endswith(s1):
            continue
        base = col[: -len(s1)]
        if base in seen:
            continue
        if base + s2 in cols_set and base + s3 in cols_set:
            bases.append(base)
            seen.add(base)

    return bases


def euler_yzx_to_matrix_deg(euler_deg: np.ndarray) -> np.ndarray:
    """
    Euler(deg, [Y, Z, X] 순서) -> 회전 행렬(3x3)
    convention = 'YZX'
    """
    euler_deg = np.asarray(euler_deg, dtype=float)
    a, b, c = np.deg2rad(euler_deg)  # Y, Z, X

    ca, cb, cc = math.cos(a), math.cos(b), math.cos(c)
    sa, sb, sc = math.sin(a), math.sin(b), math.sin(c)

    Ry = np.array(
        [
            [ca, 0.0, sa],
            [0.0, 1.0, 0.0],
            [-sa, 0.0, ca],
        ],
        dtype=float,
    )

    Rz = np.array(
        [
            [cb, -sb, 0.0],
            [sb, cb, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    Rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cc, -sc],
            [0.0, sc, cc],
        ],
        dtype=float,
    )

    # R = Ry @ Rz @ Rx (YZX 순서)
    return Ry @ Rz @ Rx


def matrix_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """
    회전 행렬(3x3) -> axis-angle 벡터(3,)  (rad)
    """
    R = np.asarray(R, dtype=float)
    trace = np.trace(R)
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = max(min(cos_theta, 1.0), -1.0)
    theta = math.acos(cos_theta)

    if theta < 1e-8:
        return np.zeros(3, dtype=float)

    denom = 2.0 * math.sin(theta)
    wx = (R[2, 1] - R[1, 2]) / denom
    wy = (R[0, 2] - R[2, 0]) / denom
    wz = (R[1, 0] - R[0, 1]) / denom
    axis = np.array([wx, wy, wz], dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-8:
        return np.zeros(3, dtype=float)
    axis = axis / norm
    return axis * theta  # rad


def euler_deg_to_axis_angle(euler_deg: np.ndarray) -> np.ndarray:
    """
    Euler(deg, YZX) -> axis-angle(rad)
    """
    R = euler_yzx_to_matrix_deg(euler_deg)
    return matrix_to_axis_angle(R)


def main():
    p = argparse.ArgumentParser(
        description="Euler(deg, YZX) CSV -> axis-angle(wx, wy, wz) CSV (모든 관절 대상)"
    )

    p.add_argument("--input", required=True, help="입력 Euler CSV 경로")
    p.add_argument("--output", required=True, help="출력 axis-angle CSV 경로")

    p.add_argument(
        "--euler-suffixes",
        nargs=3,
        default=["X", "Y", "Z"],
        metavar=("EX_SUFF", "EY_SUFF", "EZ_SUFF"),
        help="Euler(deg) 컬럼 suffix 3개 (기본: X Y Z)",
    )

    p.add_argument(
        "--aa-suffixes",
        nargs=3,
        default=["wx", "wy", "wz"],
        metavar=("WX_SUFF", "WY_SUFF", "WZ_SUFF"),
        help="axis-angle(rad) 컬럼 suffix 3개 (기본: wx wy wz)",
    )

    p.add_argument(
        "--keep-euler",
        action="store_true",
        help="출력 CSV에서 Euler 컬럼(X,Y,Z 세트)을 유지할지 여부 (기본: 제거)",
    )

    args = p.parse_args()

    df = pd.read_csv(args.input)

    # 1) Euler 컬럼 순서를 기준으로 joint base 탐색
    joint_bases = find_joint_bases(df.columns, args.euler_suffixes)
    if not joint_bases:
        raise RuntimeError("Euler 3개 세트(X/Y/Z)를 가진 관절 컬럼을 찾지 못함")

    # print("Detected joints (order preserved):")
    # for base in joint_bases:
    #     print("  ", base)

    out = df.copy()

    # 2) 관절별로 Euler -> axis-angle
    for base in joint_bases:
        euler_cols = [base + s for s in args.euler_suffixes]
        for c in euler_cols:
            if c not in out.columns:
                raise KeyError(f"Euler 컬럼 {c!r} 이(가) 없음")

        aa_list = []
        for _, row in out.iterrows():
            e_deg = row[euler_cols].to_numpy(dtype=float)
            aa = euler_deg_to_axis_angle(e_deg)  # rad
            aa_list.append(aa)

        aa_arr = np.vstack(aa_list)

        aa_cols = [base + s for s in args.aa_suffixes]
        for i, c in enumerate(aa_cols):
            out[c] = aa_arr[:, i]

    # 3) 필요하면 Euler 컬럼 제거
    if not args.keep_euler:
        drop_cols = []
        for base in joint_bases:
            for s in args.euler_suffixes:
                col = base + s
                if col in out.columns:
                    drop_cols.append(col)
        out = out.drop(columns=drop_cols)

    # 4) 컬럼 순서 재정렬 (예: Frame, Time, 그 다음 각 관절별 px,py,pz,wx,wy,wz)
    cols = list(out.columns)

    # Frame / Time 같은 공용 컬럼 먼저 유지
    prefix_cols = [c for c in cols if c.lower() in ("frame", "time")]

    pos_suff = ["px", "py", "pz"]
    aa_suff = list(args.aa_suffixes)  # 보통 ['wx','wy','wz']

    body_cols = []
    for base in joint_bases:
        # 해당 base에 대해 px,py,pz,wx,wy,wz 순서로 붙이기
        for s in pos_suff:
            col = base + s
            if col in out.columns:
                body_cols.append(col)
        for s in aa_suff:
            col = base + s
            if col in out.columns:
                body_cols.append(col)

    # 나머지(혹시 남은 컬럼)들
    used = set(prefix_cols) | set(body_cols)
    rest_cols = [c for c in cols if c not in used]

    pos_cols = [c for c in cols if c.endswith(("_px", "_py", "_pz"))]
    aa_cols = [c for c in cols if c.endswith(tuple(args.aa_suffixes))]
    new_cols = prefix_cols + aa_cols + pos_cols + rest_cols
    out = out[new_cols]
    print(out.columns)

    # 5) 소수점 6자리까지 저장
    out.to_csv(args.output, index=False, float_format="%.6f")
    print(f"Saved axis-angle CSV -> {args.output}")


if __name__ == "__main__":
    main()


"""
python 33euler_to_aa_csv.py \
  --input  Data/csv/11edited_s_stepover_local_1118.csv \
  --output Data/csv/33edited_s_stepover_local_1118.csv \
  --euler-suffixes ex ey ez \
  --aa-suffixes wx wy wz
"""
