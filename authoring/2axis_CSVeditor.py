# keyframe_editor.py

import argparse
from typing import List, Dict

import numpy as np
import pandas as pd


def _moving_average(a: np.ndarray, k: int) -> np.ndarray:
    """two_axis_editor에서 쓰던 방식 그대로 가져온 단순 이동평균 필터"""
    if k is None or k <= 1:
        return a
    k = int(k)
    k = min(k, len(a) if len(a) > 0 else 1)
    if k <= 1:
        return a
    pad = k // 2
    pad_left = a[:1].repeat(pad)
    pad_right = a[-1:].repeat(pad)
    ap = np.concatenate([pad_left, a, pad_right])
    kernel = np.ones(k) / k
    return np.convolve(ap, kernel, mode="valid")


def parse_keyframe_specs(
    specs: List[str], cols: List[str]
) -> Dict[int, Dict[str, float]]:
    """
    --cols R_Hip_X R_Hip_Y R_Hip_Z
    --keyframe 10:-20,5,10
    --keyframe 25:-40,0,20
    이런 식으로 들어온 걸

    -> {10: {"R_Hip_X":-20, "R_Hip_Y":5, "R_Hip_Z":10}, 25: {...}} 로 변환
    """
    out: Dict[int, Dict[str, float]] = {}
    for s in specs:
        if ":" not in s:
            raise ValueError(f"Invalid keyframe spec (missing ':'): {s}")
        f_str, vals_str = s.split(":", 1)
        f_idx = int(f_str)
        vals = [float(v) for v in vals_str.split(",") if v != ""]
        if len(vals) != len(cols):
            raise ValueError(
                f"Keyframe {s} has {len(vals)} values but {len(cols)} cols specified"
            )
        out[f_idx] = {c: v for c, v in zip(cols, vals)}
    return out


def apply_keyframe_edits(
    df: pd.DataFrame,
    keyframe_edits: Dict[int, Dict[str, float]],
    target_cols: List[str],
) -> pd.DataFrame:
    """
    keyframe_edits: {frame_idx: {col: value, ...}, ...}
    target_cols   : 보간/수정 대상 컬럼 리스트
    """
    if not keyframe_edits:
        return df.copy()
    out = df.copy()

    # 프레임 인덱스 유효성 체크
    max_idx = len(out) - 1
    for f in keyframe_edits.keys():
        if f < 0 or f > max_idx:
            raise ValueError(f"frame index {f} out of range [0, {max_idx}]")

    # 1) 키프레임 값 주입
    for f_idx, col_changes in keyframe_edits.items():
        for col, val in col_changes.items():
            if col not in out.columns:
                raise KeyError(f"Column {col} not found in CSV")
            out.loc[f_idx, col] = float(val)

    # 2) 각 컬럼별로 키프레임 사이 선형 보간
    sorted_frames = sorted(keyframe_edits.keys())
    frames_arr = np.arange(len(out))

    for col in target_cols:
        series = out[col].to_numpy(dtype=float, copy=True)

        # 이 컬럼에 대해 실제로 값이 지정된 키프레임만 사용
        col_key_frames = [f for f in sorted_frames if col in keyframe_edits[f]]
        if len(col_key_frames) < 2:
            # 두 개 미만이면 그냥 값만 덮어쓰고 끝
            out[col] = series
            continue

        for f0, f1 in zip(col_key_frames[:-1], col_key_frames[1:]):
            if f1 <= f0:
                continue
            v0 = series[f0]
            v1 = series[f1]

            seg_idx = (frames_arr >= f0) & (frames_arr <= f1)
            seg_frames = frames_arr[seg_idx]
            t = (seg_frames - f0) / (f1 - f0)
            series[seg_idx] = (1.0 - t) * v0 + t * v1

        out[col] = series

    return out


def main():
    p = argparse.ArgumentParser(
        description="Edit joint trajectories by keyframes & interpolation (CSV in/out)"
    )

    # 기본 입출력
    p.add_argument("--input", required=True, help="입력 CSV 경로")
    p.add_argument("--output", required=True, help="출력 CSV 경로")

    # 어떤 컬럼을 편집/보간할지
    p.add_argument(
        "--cols",
        nargs="+",
        required=True,
        help="편집/보간할 컬럼 이름들 (예: R_Hip_X R_Hip_Y R_Hip_Z)",
    )

    # 키프레임 여러 개
    p.add_argument(
        "--keyframe",
        action="append",
        metavar="SPEC",
        required=True,
        help="키프레임 스펙, 예: 10:-20,5,10  (frame:val_for_each_col). 여러 번 넣어서 사용",
    )

    # 선택: 보간 후 스무딩
    p.add_argument(
        "--smooth-window",
        type=int,
        default=None,
        help="보간 후 이동평균 윈도우 크기 (프레임 단위, 선택)",
    )

    args = p.parse_args()

    # CSV 로드
    df = pd.read_csv(args.input)

    # 키프레임 파싱
    keyframe_edits = parse_keyframe_specs(args.keyframe, args.cols)

    # 키프레임 적용 + 선형 보간
    out = apply_keyframe_edits(df, keyframe_edits, args.cols)

    # 필요하면 스무딩
    if args.smooth_window and args.smooth_window > 1:
        for col in args.cols:
            vals = out[col].to_numpy(dtype=float, copy=True)
            out[col] = _moving_average(vals, args.smooth_window)

    # 새 CSV 저장
    out.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

"""
python 2axis_CSVeditor.py \
  --input Data/csv/sensing_stepover_local_1118.csv \
  --output Data/csv/Edited_s_stepover_local_1118.csv \
  --cols right_hip_wx right_hip_wy right_hip_wz \
  --keyframe 10:-20,5,10 \
  --keyframe 25:-40,0,20 \
  --keyframe 40:-10,10,5 \
  --smooth-window 5
"""
