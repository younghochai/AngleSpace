import argparse
from typing import List, Dict

import numpy as np
import pandas as pd


def _moving_average(a: np.ndarray, k: int) -> np.ndarray:
    """단순 이동평균 필터 (스무딩 옵션용)"""
    if k is None or k <= 1:
        return a
    k = int(k)
    if k <= 1 or len(a) == 0:
        return a
    pad = k // 2
    pad_left = a[:1].repeat(pad)
    pad_right = a[-1:].repeat(pad)
    ap = np.concatenate([pad_left, a, pad_right])
    kernel = np.ones(k, dtype=np.float64) / float(k)
    return np.convolve(ap, kernel, mode="valid")


def parse_keyframe_specs(specs: List[str], cols: List[str]) -> Dict[int, Dict[str, float]]:
    """
    --keyframe 옵션 파싱 (linear / global-affine / scale 용)
    예) "10:-20,,30"  -> frame 10, cols[0]=-20, cols[2]=30
    빈칸("")은 해당 축은 건드리지 않음
    """
    keyframe_edits: Dict[int, Dict[str, float]] = {}
    if not specs:
        return keyframe_edits

    for spec in specs:
        if ":" not in spec:
            raise ValueError(f"keyframe 형식 잘못됨: {spec!r} (예: 10:-20,,30)")
        frame_str, values_str = spec.split(":", 1)
        frame_idx = int(frame_str)
        raw_tokens = values_str.split(",")

        if len(raw_tokens) != len(cols):
            raise ValueError(
                f"keyframe 값 개수가 cols 개수와 다름: spec={spec!r}, "
                f"len(tokens)={len(raw_tokens)}, len(cols)={len(cols)}"
            )

        col_changes: Dict[str, float] = {}
        for i, token in enumerate(raw_tokens):
            if token == "":
                continue  # 이 축은 수정 없음
            col_name = cols[i]
            col_changes[col_name] = float(token)

        if not col_changes:
            # 이 프레임에서는 아무 컬럼도 수정 안 하면 그냥 스킵
            continue

        if frame_idx not in keyframe_edits:
            keyframe_edits[frame_idx] = {}

        # 같은 프레임에 여러 번 keyframe 주면 나중 것이 덮어씀
        keyframe_edits[frame_idx].update(col_changes)

    return keyframe_edits


# --------------------------
# mode=linear  (기존 동작)
# --------------------------
def apply_keyframe_edits_linear(
    df: pd.DataFrame,
    target_cols: List[str],
    keyframe_edits: Dict[int, Dict[str, float]],
) -> pd.DataFrame:
    """
    기존 방식: 각 컬럼별로 키프레임 사이를 선형보간해서 덮어쓰는 모드
    """
    out = df.copy()
    n = len(out)
    if n == 0 or not keyframe_edits:
        return out

    # 1) 키프레임 프레임에 값 먼저 찍기
    for f_idx, col_changes in keyframe_edits.items():
        if f_idx < 0 or f_idx >= n:
            raise IndexError(f"keyframe frame index {f_idx} 가 범위를 벗어남 (0 ~ {n-1})")
        for col, v in col_changes.items():
            if col not in out.columns:
                raise KeyError(f"컬럼 {col!r} 이 DataFrame에 없음")
            out.at[f_idx, col] = v

    # 2) 각 컬럼마다, 자신의 키프레임들 사이를 선형보간
    frames_arr = np.arange(n, dtype=np.int64)
    sorted_frames = sorted(keyframe_edits.keys())

    for col in target_cols:
        if col not in out.columns:
            raise KeyError(f"cols 에 지정했지만 DataFrame에 없는 컬럼: {col!r}")

        series = out[col].to_numpy(dtype=np.float64, copy=True)
        # 이 컬럼에 실제 값이 지정된 키프레임만 모음
        col_key_frames = [f for f in sorted_frames if col in keyframe_edits[f]]

        # 키프레임이 2개 미만이면, 보간할 구간이 없음 -> 그냥 그대로 둠
        if len(col_key_frames) < 2:
            out[col] = series
            continue

        for f0, f1 in zip(col_key_frames[:-1], col_key_frames[1:]):
            v0 = series[f0]
            v1 = series[f1]
            if f1 == f0:
                continue
            seg_mask = (frames_arr >= f0) & (frames_arr <= f1)
            seg_frames = frames_arr[seg_mask]
            # 0~1 사이 비율
            t = (seg_frames - f0).astype(np.float64) / float(f1 - f0)
            series[seg_mask] = (1.0 - t) * v0 + t * v1

        out[col] = series

    return out


# ------------------------------
# mode=global-affine
# ------------------------------
def apply_keyframe_edits_global_affine(
    df: pd.DataFrame,
    target_cols: List[str],
    keyframe_edits: Dict[int, Dict[str, float]],
) -> pd.DataFrame:
    """
    새 방식: 키프레임을 '지표'로 사용해서
    각 컬럼마다 전역 선형 변환 y' = a * y + b 를 추정해서 전체에 적용하는 모드.

    - 프레임 사이 값을 직접 선형보간해서 갈아끼우지 않음
    - 원래 곡선 모양을 최대한 유지하면서, 키프레임에서의 값은
      (가능한 한) 내가 지정한 값에 가깝게 맞추는 느낌
    """
    out = df.copy()
    n = len(out)
    if n == 0 or not keyframe_edits:
        return out

    # 키프레임 프레임 인덱스 유효성 체크
    for f_idx in keyframe_edits.keys():
        if f_idx < 0 or f_idx >= n:
            raise IndexError(f"keyframe frame index {f_idx} 가 범위를 벗어남 (0 ~ {n-1})")

    sorted_frames = sorted(keyframe_edits.keys())

    for col in target_cols:
        if col not in out.columns:
            raise KeyError(f"cols 에 지정했지만 DataFrame에 없는 컬럼: {col!r}")

        series = out[col].to_numpy(dtype=np.float64, copy=True)

        # 이 컬럼에 대해 실제로 값이 지정된 키프레임만 모음
        xs = []  # 원본 값들
        ys = []  # 목표 값들
        for f_idx in sorted_frames:
            col_changes = keyframe_edits[f_idx]
            if col in col_changes:
                xs.append(series[f_idx])
                ys.append(col_changes[col])

        if len(xs) == 0:
            # 이 컬럼은 키프레임 지표가 없음 -> 그대로 둠
            continue

        x = np.asarray(xs, dtype=np.float64)
        y = np.asarray(ys, dtype=np.float64)

        # 키프레임이 1개면: 스케일은 유지(a=1)하고 전체 오프셋만 맞춰줌
        if len(x) == 1:
            a = 1.0
            b = float(y[0] - x[0])
        else:
            # 최소제곱으로 y ≈ a * x + b 풀기
            A = np.stack([x, np.ones_like(x)], axis=1)  # shape (m, 2)
            try:
                sol, *_ = np.linalg.lstsq(A, y, rcond=None)
                a, b = float(sol[0]), float(sol[1])
            except np.linalg.LinAlgError:
                # 특이한 경우엔 대충 평균 맞추는 오프셋만 적용
                a = 1.0
                b = float(y.mean() - x.mean())

        transformed = a * series + b
        out[col] = transformed

    return out


# ------------------------------
# mode=scale
# ------------------------------
def apply_keyframe_edits_scale(
    df: pd.DataFrame,
    keyframe_edits: Dict[int, Dict[str, float]],
    target_cols: List[str],
    anchor: float = 0.0,
) -> pd.DataFrame:
    """
    키프레임 값을 '타겟 값'으로 쓰되,
    구간 사이에서는 원본 곡선에 스케일 팩터를 선형 보간해서 적용하는 버전
    """
    if not keyframe_edits:
        return df.copy()

    out = df.copy()
    n = len(out)
    frames_arr = np.arange(n)

    for col in target_cols:
        # 이 컬럼에 실제로 키프레임이 있는 프레임만 추출
        col_key_frames = sorted(
            f for f in keyframe_edits.keys() if col in keyframe_edits[f]
        )

        # 키프레임이 없으면 원본 그대로
        if not col_key_frames:
            continue

        orig = df[col].to_numpy(dtype=float)
        new = orig.copy()

        # 스케일 factor 배열 (기본 1.0 = 변화 없음)
        scales = np.ones_like(orig, dtype=float)

        # 1) 키프레임 위치에서 scale 계산
        for f in col_key_frames:
            v_orig = orig[f]
            v_target = keyframe_edits[f][col]
            denom = (v_orig - anchor)

            if abs(denom) < 1e-8:
                # anchor랑 같으면 스케일 의미 없어짐 → 그 프레임만 나중에 강제 덮어쓰기
                scales[f] = 1.0
                new[f] = v_target
            else:
                scales[f] = (v_target - anchor) / denom

        # 2) 키프레임이 2개 이상일 때, 사이 구간에 대해 scale 선형 보간
        if len(col_key_frames) >= 2:
            for f0, f1 in zip(col_key_frames[:-1], col_key_frames[1:]):
                s0, s1 = scales[f0], scales[f1]

                seg_mask = (frames_arr >= f0) & (frames_arr <= f1)
                seg_frames = frames_arr[seg_mask]

                if f1 == f0:
                    continue

                t = (seg_frames - f0) / (f1 - f0)
                scales[seg_mask] = (1.0 - t) * s0 + t * s1

        # 3) 실제로 스케일 적용
        new = anchor + scales * (orig - anchor)

        # 4) 키프레임 프레임은 정확하게 타겟 값으로 덮어쓰기
        for f in col_key_frames:
            new[f] = keyframe_edits[f][col]

        out[col] = new

    return out


# ------------------------------
# mode=pivot  (신규)
# ------------------------------
def apply_pivot_segment_1d(
    values: np.ndarray,
    fs: int,
    ft: int,
    fe: int,
    v_target: float,
) -> np.ndarray:
    """
    1D 배열에 대해 피벗 보정 적용:
      - fs, fe: 구간 양 끝 프레임 (값 유지)
      - ft: 피벗 프레임 (값을 v_target으로 맞춤)
      - 그 사이 프레임은 삼각형 가중치로 delta 분배

    values 인덱스 == 프레임 인덱스라고 가정.
    """
    if not (fs <= ft <= fe):
        raise ValueError(f"pivot indices must satisfy fs <= ft <= fe (got {fs}, {ft}, {fe})")
    if fs < 0 or fe >= len(values):
        raise IndexError(f"pivot range [{fs}, {fe}] is out of bounds for length {len(values)}")

    out = values.copy()

    v_orig_t = values[ft]
    delta = v_target - v_orig_t

    # ft == fs 또는 ft == fe인 극단 케이스도 안전하게 처리
    left_len = max(ft - fs, 1)
    right_len = max(fe - ft, 1)

    for i in range(fs, fe + 1):
        if i < ft:
            # fs → ft로 갈수록 0 → 1
            w = (i - fs) / left_len
        elif i > ft:
            # ft → fe로 갈수록 1 → 0
            w = (fe - i) / right_len
        else:
            # i == ft
            w = 1.0

        out[i] = values[i] + w * delta

    return out


def parse_pivot_keyframe_spec(spec: str):
    """
    pivot 전용 keyframe 파서.

    형식:
      fs-ft-fe:val_x,val_y,val_z

    예:
      '10-20-30:15,,'   -> fs=10, ft=20, fe=30, axis_idx=0(x), v_target=15
      '10-20-30:,15,'   -> axis_idx=1(y)
      '10-20-30:,,15'   -> axis_idx=2(z)

    한 spec 안에서 정확히 한 축만 값이 있어야 함.
    """
    if ":" not in spec:
        raise ValueError(
            f"pivot keyframe 형식 잘못됨: {spec!r} "
            f"(예: 10-20-30:,,15)"
        )

    frame_part, value_part = spec.split(":", 1)
    try:
        fs_str, ft_str, fe_str = frame_part.split("-")
        fs, ft, fe = int(fs_str), int(ft_str), int(fe_str)
    except ValueError:
        raise ValueError(f"pivot frame part must be 'fs-ft-fe' (got: {frame_part!r})")

    vals = value_part.split(",")
    if len(vals) != 3:
        raise ValueError(
            f"pivot value part must have 3 comma-separated entries "
            f"(got: {value_part!r})"
        )

    axis_index = None
    v_target = None

    for idx, v_str in enumerate(vals):
        v_str = v_str.strip()
        if v_str == "":
            continue
        if axis_index is not None:
            # 두 축 이상에 값 넣으면 안 됨 (단일축 전용)
            raise ValueError(
                f"pivot supports only single axis per keyframe "
                f"(got multiple values in: {spec!r})"
            )
        axis_index = idx
        v_target = float(v_str)

    if axis_index is None:
        raise ValueError(
            f"pivot keyframe must specify exactly one axis value (got: {spec!r})"
        )

    return fs, ft, fe, axis_index, v_target


def apply_keyframe_edits_pivot(
    df: pd.DataFrame,
    cols: List[str],
    pivot_specs: List[str],
) -> pd.DataFrame:
    """
    pivot 모드:
      - 각 keyframe spec마다 (fs, ft, fe, axis, target) 파싱
      - 해당 축 컬럼만 fs~fe 구간에 피벗 보정 적용
    """
    out = df.copy()
    if not pivot_specs:
        return out

    n = len(out)

    for spec in pivot_specs:
        fs, ft, fe, axis_idx, v_target = parse_pivot_keyframe_spec(spec)

        if axis_idx >= len(cols):
            raise ValueError(
                f"pivot axis index {axis_idx} (from spec {spec!r}) "
                f"out of cols range (len={len(cols)})"
            )

        col_name = cols[axis_idx]
        if col_name not in out.columns:
            raise KeyError(
                f"pivot 대상 컬럼 {col_name!r} 이 DataFrame에 없음 "
                f"(spec={spec!r})"
            )

        values = out[col_name].to_numpy(dtype=float, copy=True)

        # 인덱스 == 프레임 인덱스라고 가정
        if fs < 0 or fe >= n:
            raise IndexError(
                f"pivot range [{fs}, {fe}] is out of bounds for length {n} "
                f"(spec={spec!r})"
            )

        new_values = apply_pivot_segment_1d(values, fs, ft, fe, v_target)
        out[col_name] = new_values

    return out


# ------------------------------
# main
# ------------------------------
def main() -> None:
    p = argparse.ArgumentParser(
        description="Euler(deg) joint trajectories keyframe editor (CSV in/out)"
    )

    p.add_argument("--input", required=True, help="입력 Euler CSV 경로")
    p.add_argument("--output", required=True, help="출력 Euler CSV 경로")

    p.add_argument(
        "--cols",
        nargs="+",
        required=True,
        help="편집/보간할 Euler 컬럼 이름들 (예: right_hip_ex right_hip_ey right_hip_ez)",
    )

    p.add_argument(
        "--keyframe",
        nargs="+",
        help=(
            "키프레임 지정.\n"
            "  linear/global-affine/scale: frame:val_x,val_y,val_z (예: 10:-20,,30)\n"
            "  pivot: fs-ft-fe:val_x,val_y,val_z (예: 10-20-30:,,15)"
        ),
    )

    p.add_argument(
        "--mode",
        choices=["linear", "global-affine", "scale", "pivot"],
        default="linear",
        help=(
            "linear: 키프레임 사이 선형 보간\n"
            "global-affine: 전역 y'=a*y+b 추정\n"
            "scale: anchor 기준 스케일 보간\n"
            "pivot: fs-ft-fe 구간에서 피벗 프레임 기준 삼각형 보정"
        ),
    )

    p.add_argument(
        "--smooth-window",
        type=int,
        default=None,
        help="보간/변환 후 이동평균 스무딩 윈도우 (프레임 단위)",
    )

    args = p.parse_args()

    df = pd.read_csv(args.input)

    cols: List[str] = list(args.cols)

    # mode별 keyframe 파싱/적용
    if args.mode in ("linear", "global-affine", "scale"):
        keyframe_edits = parse_keyframe_specs(args.keyframe or [], cols)

        if args.mode == "linear":
            out = apply_keyframe_edits_linear(df, cols, keyframe_edits)
        elif args.mode == "global-affine":
            out = apply_keyframe_edits_global_affine(df, cols, keyframe_edits)
        elif args.mode == "scale":
            out = apply_keyframe_edits_scale(df, keyframe_edits, cols)
        else:
            raise ValueError(f"Unknown mode: {args.mode}")
    elif args.mode == "pivot":
        out = apply_keyframe_edits_pivot(df, cols, args.keyframe or [])
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    # 옵션 스무딩
    if args.smooth_window and args.smooth_window > 1:
        for col in cols:
            if col not in out.columns:
                continue
            vals = out[col].to_numpy(dtype=np.float64, copy=True)
            out[col] = _moving_average(vals, args.smooth_window)

    out.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()



'''
python 22keyframe_editor_euler.py \
  --input Data/csv/11edited_s_stepover_local_1118.csv \
  --output Data/csv/22edited_s_stepover_local_1118.csv \
  --cols right_hip_ex right_hip_ey right_hip_ez \
  --keyframe 10:-20,, \
  --keyframe 25:-40,0,20 \
  --smooth-window 5
  --mode global-affine
'''

# python 22keyframe_editor_euler.py \
#   --input  in.csv \
#   --output out.csv \
#   --cols right_hip_ex right_hip_ey right_hip_ez \
#   --keyframe 10:-20,, \
#   --keyframe 25:-40,0,20 \
#   --mode global-affine \
#   --smooth-window 5

# python 22keyframe_editor_euler.py \
#   --input  in.csv \
#   --output out.csv \
#   --cols right_hip_ex right_hip_ey right_hip_ez \
#   --mode pivot \
#   --keyframe 10-20-30:,,15
