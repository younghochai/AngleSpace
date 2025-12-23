# aa_to_euler_csv.py
import argparse
import numpy as np
import pandas as pd

from utils import convert_pose_to_euler  # axis-angle(3,) -> euler_deg(3,)


def find_joint_bases(columns, aa_suffixes):
    s1, s2, s3 = aa_suffixes
    cols_set = set(columns)
    bases = []
    seen = set()

    for col in columns:
        if not col.endswith(s1):
            continue
        base = col[: -len(s1)]
        if base in seen:
            continue
        # 같은 base에 대해 s2, s3도 있는지 체크
        if base + s2 in cols_set and base + s3 in cols_set:
            bases.append(base)
            seen.add(base)

    return bases


def main():
    p = argparse.ArgumentParser(
        description="Axis-angle(wx, wy, wz) CSV -> Euler(deg) CSV (모든 관절 대상)"
    )
    p.add_argument("--input", required=True, help="입력 axis-angle CSV 경로")
    p.add_argument("--output", required=True, help="출력 Euler CSV 경로")

    p.add_argument(
        "--aa-suffixes",
        nargs=3,
        default=["wx", "wy", "wz"],
        metavar=("WX_SUFF", "WY_SUFF", "WZ_SUFF"),
        help="axis-angle 컬럼 suffix 3개 (기본: wx wy wz)",
    )

    p.add_argument(
        "--euler-suffixes",
        nargs=3,
        default=["X", "Y", "Z"],
        metavar=("EX_SUFF", "EY_SUFF", "EZ_SUFF"),
        help="Euler(deg) 컬럼 suffix 3개 (기본: X Y Z)",
    )

    p.add_argument(
        "--drop-aa",
        action="store_true",
        help="출력 CSV에서 axis-angle(wx,wy,wz 세트)을 제거할지 여부",
    )

    args = p.parse_args()

    df = pd.read_csv(args.input)

    # 1) 원래 CSV 열 순서를 기준으로 joint base 탐색
    joint_bases = find_joint_bases(df.columns, args.aa_suffixes)
    if not joint_bases:
        raise RuntimeError("axis-angle 3개 세트(wx/wy/wz)를 가진 관절 컬럼을 찾지 못함")

    # print("Detected joints (order preserved):")
    # for base in joint_bases:
    #     print("  ", base)

    out = df.copy()

    # 2) 관절별로 axis-angle -> Euler
    for base in joint_bases:
        aa_cols = [base + s for s in args.aa_suffixes]
        for c in aa_cols:
            if c not in out.columns:
                raise KeyError(f"axis-angle 컬럼 {c!r} 이(가) 없음")

        euler_list = []
        for _, row in out.iterrows():
            aa_vec = row[aa_cols].to_numpy(dtype=float)  # rad 가정
            e_deg = convert_pose_to_euler(aa_vec)        # np.array(3,) deg
            euler_list.append(e_deg)

        e_arr = np.vstack(euler_list)  # (T,3)

        euler_cols = [base + s for s in args.euler_suffixes]
        for i, c in enumerate(euler_cols):
            out[c] = e_arr[:, i]

    # 3) 필요하면 axis-angle 컬럼 드롭
    if args.drop_aa:
        drop_cols = []
        for base in joint_bases:
            for s in args.aa_suffixes:
                col = base + s
                if col in out.columns:
                    drop_cols.append(col)
        out = out.drop(columns=drop_cols)

    out.to_csv(args.output, index=False)
    print(f"Saved Euler CSV -> {args.output}")


if __name__ == "__main__":
    main()


'''
python 11aa_to_euler_csv.py \
  --input  Data/csv/sensing_stepover_local_1118.csv \
  --output Data/csv/11edited_s_stepover_local_1118.csv \
  --aa-suffixes wx wy wz \
  --euler-suffixes ex ey ez \
  --drop-aa
'''

# python aa_to_euler_csv11.py \
#   --input  Data/csv/edited_1axis_s_SO_local_1204.csv \
#   --output Data/csv/ee_edited_1axis_s_SO_local_1204.csv \
#   --aa-suffixes wx wy wz \
#   --euler-suffixes ex ey ez \
#   --drop-aa