import argparse
from typing import Optional, Tuple
import numpy as np
import pandas as pd


def _moving_average(a: np.ndarray, k: int) -> np.ndarray:
    """Simple centered moving average for 1D array."""
    if k is None or k <= 1:
        return a
    k = int(k)
    if len(a) == 0:
        return a
    k = min(k, len(a))
    if k <= 1:
        return a
    pad = k // 2
    # reflect padding to avoid edge shrink
    padded = np.pad(a, (pad, pad), mode="edge")
    kernel = np.ones(k, dtype=float) / float(k)
    smoothed = np.convolve(padded, kernel, mode="same")
    return smoothed[pad:-pad]


def _slice_index_by_time_or_frame(
    df: pd.DataFrame,
    frame_range: Optional[Tuple[int, int]],
    time_range: Optional[Tuple[float, float]],
) -> np.ndarray:
    """Return indices of rows selected by either frame_range or time_range."""
    if frame_range is not None and time_range is not None:
        raise ValueError("Provide only one of frame-range or time-range")

    if frame_range is not None:
        i0, i1 = frame_range
        i0 = max(0, int(i0))
        i1 = min(len(df) - 1, int(i1))
        if i1 < i0:
            raise ValueError("frame-range end < start")
        return np.arange(i0, i1 + 1)

    if time_range is not None:
        if "Time" not in df.columns:
            raise ValueError("Time column not found for time-range selection")
        t0, t1 = time_range
        if t1 < t0:
            raise ValueError("time-range end < start")
        mask = (df["Time"] >= float(t0)) & (df["Time"] <= float(t1))
        return np.where(mask.values)[0]

    # default: all
    return np.arange(len(df))


def _get_col(joint: str, axis: str) -> str:
    """Build column name from joint and axis, e.g. 'right_hip', 'wy' -> 'right_hip_wy'."""
    joint = joint.strip()
    axis = axis.strip()
    if not joint:
        raise ValueError("joint must be non-empty")
    if not axis:
        raise ValueError("axis must be non-empty")
    return f"{joint}_{axis}"


def _anchor_value(values: np.ndarray, kind: str) -> float:
    """Anchor point for 1D scaling."""
    kind = (kind or "origin").lower()
    if kind == "origin":
        return 0.0
    elif kind == "mean":
        return float(values.mean())
    elif kind == "start":
        return float(values[0])
    elif kind == "end":
        return float(values[-1])
    else:
        raise ValueError("anchor must be one of origin/mean/start/end")


def apply_translate_1d(values: np.ndarray, dt: float) -> np.ndarray:
    return values + float(dt)


def apply_scale_1d(values: np.ndarray, scale: float, anchor: str) -> np.ndarray:
    a = _anchor_value(values, anchor)
    return (values - a) * float(scale) + a


def edit_1axis_trajectory(
    df: pd.DataFrame,
    joint: str,
    axis: str,
    frame_range: Optional[Tuple[int, int]] = None,
    time_range: Optional[Tuple[float, float]] = None,
    op: str = "translate",
    dt: float = 0.0,
    scale: float = 1.0,
    anchor: str = "origin",
) -> pd.DataFrame:
    """
    1D 궤적 편집:
      - translate: 지정 구간에 dt 더하기
      - scale: anchor 기준으로 scale 배
    """
    col = _get_col(joint, axis)
    if col not in df.columns:
        raise KeyError(f"Column not found: {col}")

    idx = _slice_index_by_time_or_frame(df, frame_range, time_range)
    if len(idx) == 0:
        # nothing to do
        return df.copy()

    out = df.copy()
    values = out.loc[idx, col].to_numpy(copy=True)

    op = op.lower()
    if op == "translate":
        values2 = apply_translate_1d(values, dt)
    elif op == "scale":
        values2 = apply_scale_1d(values, scale, anchor)
    else:
        raise ValueError("op must be one of: translate, scale")

    out.loc[idx, col] = values2
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="1-axis trajectory editor (translate / scale).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, help="Input CSV path")
    p.add_argument("--output", required=True, help="Output CSV path")

    p.add_argument("--joint", required=True, help="Joint name prefix, e.g. right_hip")
    p.add_argument("--axis", required=True, help="Axis suffix, e.g. wx / wy / wz / ex / ey / ez")

    p.add_argument(
        "--frame-range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Frame range [START, END] (inclusive) to edit",
    )
    p.add_argument(
        "--time-range",
        nargs=2,
        type=float,
        metavar=("T0", "T1"),
        help="Time range [T0, T1] in seconds (requires 'Time' column)",
    )

    p.add_argument(
        "--op",
        choices=["translate", "scale"],
        required=True,
        help="Operation type",
    )

    # translate
    p.add_argument(
        "--dt",
        type=float,
        default=0.0,
        help="Offset to add for translate",
    )

    # scale
    p.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Scale factor for scale op",
    )
    p.add_argument(
        "--anchor",
        choices=["origin", "mean", "start", "end"],
        default="origin",
        help="Anchor type for scale op",
    )

    p.add_argument(
        "--smooth-window",
        type=int,
        default=0,
        help="Optional moving-average window after edit (0 or 1 disables)",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.frame_range is not None and args.time_range is not None:
        raise SystemExit("Use only one of --frame-range or --time-range")

    df = pd.read_csv(args.input)

    out = edit_1axis_trajectory(
        df=df,
        joint=args.joint,
        axis=args.axis,
        frame_range=tuple(args.frame_range) if args.frame_range else None,  # type: ignore
        time_range=tuple(args.time_range) if args.time_range else None,  # type: ignore
        op=args.op,
        dt=args.dt,
        scale=args.scale,
        anchor=args.anchor,
    )

    col = _get_col(args.joint, args.axis)
    if args.smooth_window and args.smooth_window > 1:
        idx = _slice_index_by_time_or_frame(
            out,
            frame_range=tuple(args.frame_range) if args.frame_range else None,  # type: ignore
            time_range=tuple(args.time_range) if args.time_range else None,  # type: ignore
        )
        out.loc[idx, col] = _moving_average(
            out.loc[idx, col].to_numpy(copy=True),
            args.smooth_window,
        )

    out.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()


'''
python one_axis_editor.py \
  --input Data/csv/edited_s_SO_local_1204.csv \
  --output Data/csv/e_edited_s_SO_local_1204.csv \
  --joint right_hip \
  --axis ex \
  --frame-range 0 29 \
  --op translate \
  --dt -5

  
python one_axis_editor.py \
  --input Data/csv/edited_s_SO_local_1204.csv \
  --output Data/csv/1axis_s_SO_local_1204.csv \
  --joint right_hip_ex \
  --axis ex \
  --frame-range 0 60 \
  --op scale \
  --scale 0.6 \
  --anchor mean \
  --smooth-window 5
'''