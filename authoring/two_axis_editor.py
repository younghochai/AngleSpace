import argparse
import math
from typing import Tuple, Optional

import numpy as np
import pandas as pd


def _moving_average(a: np.ndarray, k: int) -> np.ndarray:
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


def _get_cols(joint: str, axes: Tuple[str, str]) -> Tuple[str, str]:
    a, b = axes
    valid = {"wx", "wy", "wz"}
    if a not in valid or b not in valid or a == b:
        raise ValueError(f"axes must be two of wx/wy/wz and distinct, got: {axes}")
    return f"{joint}_{a}", f"{joint}_{b}"


def _slice_index_by_time_or_frame(
    df: pd.DataFrame,
    frame_range: Optional[Tuple[int, int]],
    time_range: Optional[Tuple[float, float]],
) -> np.ndarray:
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


def _anchor_point(x: np.ndarray, y: np.ndarray, kind: str) -> Tuple[float, float]:
    kind = (kind or "origin").lower()
    if kind == "origin":
        return 0.0, 0.0
    elif kind == "mean":
        return float(x.mean()), float(y.mean())
    elif kind == "start":
        return float(x[0]), float(y[0])
    elif kind == "end":
        return float(x[-1]), float(y[-1])
    else:
        raise ValueError("anchor must be one of origin/mean/start/end")


def apply_translate(
    x: np.ndarray, y: np.ndarray, tx: float, ty: float
) -> Tuple[np.ndarray, np.ndarray]:
    return x + float(tx), y + float(ty)


def apply_scale(
    x: np.ndarray, y: np.ndarray, sx: float, sy: Optional[float], anchor: str
) -> Tuple[np.ndarray, np.ndarray]:
    if sy is None:
        sy = sx
    cx, cy = _anchor_point(x, y, anchor)
    x2 = (x - cx) * float(sx) + cx
    y2 = (y - cy) * float(sy) + cy
    return x2, y2


def apply_rotate(
    x: np.ndarray, y: np.ndarray, deg: float, anchor: str
) -> Tuple[np.ndarray, np.ndarray]:
    rad = math.radians(float(deg))
    cosr, sinr = math.cos(rad), math.sin(rad)
    cx, cy = _anchor_point(x, y, anchor)
    xr, yr = x - cx, y - cy
    x2 = xr * cosr - yr * sinr + cx
    y2 = xr * sinr + yr * cosr + cy
    return x2, y2


def apply_polar(
    x: np.ndarray,
    y: np.ndarray,
    dtheta_deg: float = 0.0,
    r_scale: float = 1.0,
    r_offset: float = 0.0,
    anchor: str = "origin",
) -> Tuple[np.ndarray, np.ndarray]:
    cx, cy = _anchor_point(x, y, anchor)
    xr, yr = x - cx, y - cy
    r = np.hypot(xr, yr)
    theta = np.arctan2(yr, xr) + math.radians(float(dtheta_deg))
    r2 = r * float(r_scale) + float(r_offset)
    x2 = r2 * np.cos(theta) + cx
    y2 = r2 * np.sin(theta) + cy
    return x2, y2


def edit_2axis_trajectory(
    df: pd.DataFrame,
    joint: str,
    axes: Tuple[str, str],
    frame_range: Optional[Tuple[int, int]] = None,
    time_range: Optional[Tuple[float, float]] = None,
    op: str = "translate",
    # translate
    tx: float = 0.0,
    ty: float = 0.0,
    # scale
    sx: float = 1.0,
    sy: Optional[float] = None,
    # rotate
    rotate_deg: float = 0.0,
    # polar
    dtheta_deg: float = 0.0,
    r_scale: float = 1.0,
    r_offset: float = 0.0,
    # common
    anchor: str = "origin",
) -> pd.DataFrame:
    """
    df: 원본 DataFrame
    joint: 'right_hip' 같은 조인트 이름
    axes: ('wy', 'wz') 처럼 두 축
    frame_range/time_range: 편집할 구간 선택(둘 중 하나만)
    op: 'translate' | 'scale' | 'rotate' | 'polar'
    """
    col_x, col_y = _get_cols(joint, axes)
    if col_x not in df.columns or col_y not in df.columns:
        raise KeyError(f"Columns not found: {col_x}, {col_y}")

    idx = _slice_index_by_time_or_frame(df, frame_range, time_range)
    x = df.loc[idx, col_x].to_numpy(copy=True)
    y = df.loc[idx, col_y].to_numpy(copy=True)

    op = op.lower()
    if op == "translate":
        x2, y2 = apply_translate(x, y, tx, ty)
    elif op == "scale":
        x2, y2 = apply_scale(x, y, sx, sy, anchor)
    elif op == "rotate":
        x2, y2 = apply_rotate(x, y, rotate_deg, anchor)
    elif op == "polar":
        x2, y2 = apply_polar(
            x,
            y,
            dtheta_deg=dtheta_deg,
            r_scale=r_scale,
            r_offset=r_offset,
            anchor=anchor,
        )
    else:
        raise ValueError("Unsupported op. Use translate/scale/rotate/polar")

    out = df.copy()
    out.loc[idx, col_x] = x2
    out.loc[idx, col_y] = y2
    return out


def main():
    p = argparse.ArgumentParser(description="Edit a joint's two-axis trajectory in CSV")

    p.add_argument("--output", required=True, help="output CSV path")
    p.add_argument("--joint", required=True, help="joint name, e.g., right_hip")
    p.add_argument(
        "--axes",
        nargs=2,
        required=True,
        choices=["wx", "wy", "wz"],
        help="pick two axes",
    )
    grp = p.add_mutually_exclusive_group()
    grp.add_argument(
        "--frame-range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="inclusive frame indices",
    )
    grp.add_argument(
        "--time-range",
        nargs=2,
        type=float,
        metavar=("T0", "T1"),
        help="seconds inclusive",
    )

    p.add_argument(
        "--op", required=True, choices=["translate", "scale", "rotate", "polar"]
    )
    # translate
    p.add_argument("--tx", type=float, default=0.0, help="translate x amount")
    p.add_argument("--ty", type=float, default=0.0, help="translate y amount")
    # scale
    p.add_argument("--sx", type=float, default=1.0, help="scale x")
    p.add_argument("--sy", type=float, default=None, help="scale y (default sx)")
    # rotate
    p.add_argument(
        "--rotate-deg", type=float, default=0.0, help="rotate degrees (+CCW)"
    )
    # polar
    p.add_argument(
        "--dtheta-deg", type=float, default=0.0, help="angle offset in degrees (polar)"
    )
    p.add_argument("--r-scale", type=float, default=1.0, help="radius scale (polar)")
    p.add_argument("--r-offset", type=float, default=0.0, help="radius offset (polar)")
    # common
    p.add_argument("--input", required=True, help="input CSV path")
    p.add_argument(
        "--anchor",
        type=str,
        default="origin",
        choices=["origin", "mean", "start", "end"],
        help="transform anchor",
    )
    p.add_argument(
        "--smooth-window",
        type=int,
        default=None,
        help="moving average window (frames) applied after transform",
    )

    args = p.parse_args()

    df = pd.read_csv(args.input)

    out = edit_2axis_trajectory(
        df=df,
        joint=args.joint,
        axes=tuple(args.axes),  # type: ignore
        frame_range=tuple(args.frame_range) if args.frame_range else None,  # type: ignore
        time_range=tuple(args.time_range) if args.time_range else None,  # type: ignore
        op=args.op,
        tx=args.tx,
        ty=args.ty,
        sx=args.sx,
        sy=args.sy,
        rotate_deg=args.rotate_deg,
        dtheta_deg=args.dtheta_deg,
        r_scale=args.r_scale,
        r_offset=args.r_offset,
        anchor=args.anchor,
    )
    col_x, col_y = _get_cols(args.joint, tuple(args.axes))
    if args.smooth_window:
        idx = _slice_index_by_time_or_frame(
            out,
            tuple(args.frame_range) if args.frame_range else None,
            tuple(args.time_range) if args.time_range else None,
        )
        out.loc[idx, col_x] = _moving_average(
            out.loc[idx, col_x].to_numpy(copy=True), args.smooth_window
        )
        out.loc[idx, col_y] = _moving_average(
            out.loc[idx, col_y].to_numpy(copy=True), args.smooth_window
        )
    out.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
