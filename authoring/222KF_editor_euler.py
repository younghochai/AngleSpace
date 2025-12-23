import argparse
from typing import List, Dict

import math
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


def parse_keyframe_specs(
    specs: List[str], cols: List[str]
) -> Dict[int, Dict[str, float]]:
    """
    --keyframe 옵션 파싱 (linear / global-affine / scale / local 용)
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


# ============================================================
# local 모드: 원본 그래프 shape 유지하면서 특정 프레임 주변만 수정
# ============================================================
def apply_local_gaussian_bump_1d(
    values: np.ndarray,
    center_frame: int,
    target_value: float,
    radius: int = 10,
) -> np.ndarray:
    """
    1D 배열에서 center_frame 값을 target_value로 맞추면서
    center_frame 주변 (±radius 프레임)에만 가우시안 bump를 더해주는 로컬 편집.

    - values: 원본 값 (len == 전체 프레임 수)
    - center_frame: 수정하고 싶은 프레임 인덱스
    - target_value: 해당 프레임에서의 새 값
    - radius: 영향 줄 프레임 수 (반경)
    """
    out = values.astype(float).copy()
    n = len(out)

    c = int(center_frame)
    if c < 0 or c >= n:
        raise IndexError(f"center_frame {c} out of range (0 ~ {n-1})")

    orig = out[c]
    delta = float(target_value) - orig
    if abs(delta) < 1e-8:
        # 거의 변화 없으면 그냥 반환
        return out

    if radius <= 0:
        # radius 0이면 해당 프레임만 강제
        out[c] = float(target_value)
        return out

    start = max(0, c - radius)
    end = min(n - 1, c + radius)

    sigma = radius / 2.0 if radius > 0 else 1.0

    idx = np.arange(start, end + 1)
    dist = idx - c
    w = np.exp(-0.5 * (dist / sigma) ** 2)

    out[idx] += delta * w
    # 중심 프레임은 수치 오차 없이 정확히 target으로 맞춤
    out[c] = float(target_value)

    return out


def apply_keyframe_edits_local_bump(
    df: pd.DataFrame,
    cols: List[str],
    keyframe_edits: Dict[int, Dict[str, float]],
    radius: int = 10,
) -> pd.DataFrame:
    """
    local 모드:
      - keyframe 스펙: frame:val_x,val_y,val_z (linear 등과 동일)
      - 각 frame/축마다 로컬 가우시안 bump를 적용해서
        원본 곡선은 최대한 유지하면서 해당 frame 근처만 수정
    """
    out = df.copy()
    if not keyframe_edits:
        return out

    n = len(out)

    for frame_idx, col_to_val in keyframe_edits.items():
        if frame_idx < 0 or frame_idx >= n:
            raise IndexError(f"local mode frame {frame_idx} out of range (0 ~ {n-1})")

        for col_name, target_value in col_to_val.items():
            if col_name not in out.columns:
                raise KeyError(
                    f"local mode 대상 컬럼 {col_name!r} 이 DataFrame에 없음 "
                    f"(frame={frame_idx})"
                )

            values = out[col_name].to_numpy(dtype=float, copy=True)
            new_values = apply_local_gaussian_bump_1d(
                values,
                center_frame=frame_idx,
                target_value=target_value,
                radius=radius,
            )
            out[col_name] = new_values

    return out


# ============================================================
# pivot 모드
# ============================================================
def apply_pivot_segment_1d(
    values: np.ndarray,
    fs: int,
    ft: int,
    fe: int,
    v_target: float,
) -> np.ndarray:
    """
    1D 배열 구간 [fs, fe]에 대해 라플라시안 곡선 편집 적용.

      - fs: 시작 프레임 (값 고정, 원본 유지)
      - ft: 타겟 프레임 (값을 v_target으로 강제)
      - fe: 끝 프레임 (값 고정, 원본 유지)

    내부 프레임들은
        sum_i (x[i-1] - 2 x[i] + x[i+1])^2
    를 최소화하는 방향으로 조정해서,
    원래 곡선의 '굽은 정도(2차 차분)'를 최대한 유지한 채로
    fs, ft, fe 값만 맞추는 방식임.
    """

    # fs < ft < fe 조건 강제
    if not (fs < ft < fe):
        raise ValueError(
            f"pivot indices must satisfy fs < ft < fe (got {fs}, {ft}, {fe})"
        )
    if fs < 0 or fe >= len(values):
        raise IndexError(
            f"pivot range [{fs}, {fe}] is out of bounds for length {len(values)}"
        )

    # float 복사본으로 작업
    out = values.astype(float).copy()

    # 편집 구간만 잘라서 작업 (길이 m)
    seg = out[fs : fe + 1]
    m = seg.shape[0]

    # 이론상 fs < ft < fe 이므로 m >= 3 이어야 하는데,
    # 혹시 모를 이상 케이스 방어
    if m < 3:
        # 그냥 직선 보간으로 fallback
        out[fs : fe + 1] = np.linspace(seg[0], v_target, m)
        return out

    # 타겟 인덱스 (세그먼트 기준)
    j_t = ft - fs
    if not (0 < j_t < m - 1):
        raise ValueError(
            f"target frame must lie strictly inside [fs, fe] " f"(got j_t={j_t}, m={m})"
        )

    # --- 제약 조건: fs, ft, fe 값 고정 ---
    # x[0]   = 원본 fs 값
    # x[j_t] = v_target
    # x[m-1] = 원본 fe 값
    b = np.array([seg[0], v_target, seg[-1]], dtype=float)
    A = np.zeros((3, m), dtype=float)
    A[0, 0] = 1.0  # fs
    A[1, j_t] = 1.0  # ft
    A[2, m - 1] = 1.0  # fe

    # --- 라플라시안 행렬 L (2차 차분) ---
    # L x = [x[i-1] - 2 x[i] + x[i+1]] (i = 1..m-2)
    if m > 2:
        L = np.zeros((m - 2, m), dtype=float)
        for j in range(1, m - 1):
            L[j - 1, j - 1] = 1.0
            L[j - 1, j] = -2.0
            L[j - 1, j + 1] = 1.0
    else:
        L = np.zeros((0, m), dtype=float)

    # --- KKT 시스템 구성 ---
    #   minimize ||L x||^2  subject to A x = b
    #
    # [2 L^T L  A^T] [x]   = [0]
    # [A        0 ] [λ]     [b]
    mL = m
    K = np.zeros((mL + 3, mL + 3), dtype=float)

    # 2 L^T L 부분
    if L.shape[0] > 0:
        K11 = 2.0 * (L.T @ L)
    else:
        K11 = np.zeros((mL, mL), dtype=float)

    K[:mL, :mL] = K11
    K[:mL, mL:] = A.T
    K[mL:, :mL] = A
    # 오른쪽 아래 3x3 블록은 0

    rhs = np.zeros(mL + 3, dtype=float)
    rhs[mL:] = b

    try:
        sol = np.linalg.solve(K, rhs)
        x = sol[:mL]
    except np.linalg.LinAlgError:
        # 혹시 특이행렬 뜨면, 제약조건에 큰 가중치 준 LSQ로 fallback
        w = 1000.0
        M = np.vstack((L, w * A))
        rhs2 = np.concatenate((np.zeros(L.shape[0], dtype=float), w * b))
        x, *_ = np.linalg.lstsq(M, rhs2, rcond=None)

    # 편집 구간만 새 값으로 덮어쓰기
    out[fs : fe + 1] = x
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
            f"pivot keyframe 형식 잘못됨: {spec!r} " f"(예: 10-20-30:,,15)"
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
                f"pivot 대상 컬럼 {col_name!r} 이 DataFrame에 없음 " f"(spec={spec!r})"
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


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
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
            "  linear/global-affine/scale/local: frame:val_x,val_y,val_z (예: 10:-20,,30)\n"
            "  pivot: fs-ft-fe:val_x,val_y,val_z (예: 10-20-30:,,15)"
        ),
    )

    p.add_argument(
        "--mode",
        choices=["linear", "global-affine", "scale", "pivot", "local"],
        default="linear",
        help=(
            "linear: 키프레임 사이 선형 보간\n"
            "global-affine: 전역 y'=a*y+b 추정\n"
            "scale: anchor 기준 스케일 보간\n"
            "pivot: fs-ft-fe 구간에서 피벗 프레임 기준 삼각형 보정\n"
            "local: 특정 프레임 기준 가우시안 bump로 원본 곡선 형태 최대한 유지"
        ),
    )

    p.add_argument(
        "--smooth-window",
        type=int,
        default=None,
        help="보간/변환 후 이동평균 스무딩 윈도우 (프레임 단위)",
    )
    p.add_argument(
        "--local-radius",
        type=int,
        default=10,
        help="mode=local일 때 중심 프레임 주변에 영향을 줄 반경(프레임 수)",
    )

    args = p.parse_args()

    df = pd.read_csv(args.input)
    cols: List[str] = list(args.cols)

    # ---- mode별 적용 ----
    if args.mode in ("linear", "global-affine", "scale", "local"):
        keyframe_edits = parse_keyframe_specs(args.keyframe or [], cols)

        if args.mode == "linear":
            out = apply_keyframe_edits_linear(df, cols, keyframe_edits)
        elif args.mode == "global-affine":
            out = apply_keyframe_edits_global_affine(df, cols, keyframe_edits)
        elif args.mode == "scale":
            out = apply_keyframe_edits_scale(df, keyframe_edits, cols)
        elif args.mode == "local":
            out = apply_keyframe_edits_local_bump(
                df, cols, keyframe_edits, radius=args.local_radius
            )
        else:
            raise ValueError(f"Unknown mode: {args.mode}")

    elif args.mode == "pivot":
        out = apply_keyframe_edits_pivot(df, cols, args.keyframe or [])
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    # ---- 옵션 스무딩 ----
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


"""
python 22keyframe_editor_euler.py \
  --input Data/csv/11edited_s_stepover_local_1118.csv \
  --output Data/csv/22edited_s_stepover_local_1118.csv \
  --cols right_hip_ex right_hip_ey right_hip_ez \
  --keyframe 10:-20,, \
  --keyframe 25:-40,0,20 \
  --smooth-window 5
  --mode global-affine
"""

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


# python 222KF_editor_euler.py \
#   --input Data/csv/edited_s_SO_local_1204.csv \
#   --output Data/csv/e_edited_s_SO_local_1204_local.csv \
#   --cols right_hip_ex right_hip_ey right_hip_ez \
#   --mode local \
#   --local-radius 10 \
#   --keyframe 30:45,,
