"""Fair runtime comparison of nbmorph vs fastmorph vs scipy.ndimage.

Produces one figure (2x2: threads x operation) plus a CSV of the raw timings.

Fairness rules applied here:
  * one and the same input array for every method (C-contiguous uint16),
  * every method allocates its own output (no pre-allocated `out=` shortcuts),
  * numba JIT compilation and thread-pool spin-up excluded via a warm-up call,
  * thread count pinned identically for nbmorph and fastmorph,
  * best-of-N wall clock (min is the least noise-contaminated estimator),
  * outputs are diffed against the nbmorph result and the agreement level is
    reported, so bars that are not the same operation are visibly marked,
  * the effective structuring element size (in voxels) is annotated per method,
    since a 3x3x3 box and a quasi-sphere are simply not the same amount of work.

Usage:  python scripts/benchmark_comparison.py [--repeats 5] [--radius 3]
"""

import argparse
import csv
import os
import tempfile
import time
from pathlib import Path

# Use a private numba cache. The cache that numba writes into site-packages can
# end up serving wrong machine code (segfault) once several specialisations of
# the same function have accumulated; a per-run cache keeps timings honest and
# reproducible. Compilation happens in the warm-up call and is never timed.
os.environ.setdefault("NUMBA_CACHE_DIR", tempfile.mkdtemp(prefix="nbmorph_bench_cache_"))

import numpy as np
import numba
import nbmorph
import fastmorph
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# structuring elements
# --------------------------------------------------------------------------
def box(r):
    return np.ones((2 * r + 1,) * 3, bool)


def ball(r):
    g = np.arange(-r, r + 1)
    z, y, x = np.meshgrid(g, g, g, indexing="ij")
    return (z * z + y * y + x * x) <= r * r


# --------------------------------------------------------------------------
# scipy reference implementations
#
# scipy.ndimage has no multi-label morphology. The idiomatic label-safe
# formulation without an O(n_labels) python loop is:
#   erosion : a voxel keeps its label iff min == max over the footprint
#   dilation: background voxels take the max (not the mode) of the footprint
# The dilation variant is *weaker* than a mode filter, i.e. this favours scipy.
# --------------------------------------------------------------------------
def scipy_erode(a, fp):
    lo = ndi.grey_erosion(a, footprint=fp, mode="constant", cval=0)
    hi = ndi.grey_dilation(a, footprint=fp, mode="constant", cval=0)
    return np.where(lo == hi, a, 0)


def scipy_dilate(a, fp):
    return np.where(a == 0, ndi.grey_dilation(a, footprint=fp, mode="constant", cval=0), a)


def scipy_erode_iter(a, n):
    fp = box(1)
    for _ in range(n):
        a = scipy_erode(a, fp)
    return a


def scipy_dilate_iter(a, n):
    fp = box(1)
    for _ in range(n):
        a = scipy_dilate(a, fp)
    return a


# --------------------------------------------------------------------------
# effective footprint of a method: dilate a single voxel and count reached ones
# --------------------------------------------------------------------------
def effective_footprint(fn, r):
    n = 2 * r + 3
    a = np.zeros((n, n, n), np.uint16)
    a[n // 2, n // 2, n // 2] = 1
    try:
        return int((fn(a) > 0).sum())
    except Exception:
        return 0


def timeit(fn, repeats, budget=2.0):
    """Best-of-N wall clock, stopping early once `budget` seconds are spent."""
    ts = []
    t_all = time.perf_counter()
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
        if time.perf_counter() - t_all > budget and len(ts) >= 3:
            break
    return min(ts), float(np.median(ts)), len(ts)


def agreement(ref, other):
    """How does `other` relate to the nbmorph reference `ref`?"""
    if other is None or other.shape != ref.shape:
        return "n/a"
    c = (slice(4, -4),) * 3  # ignore border-handling conventions
    if np.array_equal(ref[c], other[c]):
        return "identical"
    if np.array_equal((ref[c] != 0), (other[c] != 0)):
        return "same support"  # same voxels labelled, different tie-break
    return "different"


def build_methods(img, radius, threads):
    """name -> (callable, library, operation, radius-group)"""
    p = threads
    bin_img = img > 0
    fp_ball = ball(radius)
    m = {}

    # ---- radius 1, 3x3x3 box: exactly the same operation everywhere ----
    m[("erode", "r1", "nbmorph")] = lambda: nbmorph.erode_labels_spherical(img, 1, struct_sequence="B")
    m[("erode", "r1", "fastmorph")] = lambda: fastmorph.erode(img, parallel=p)
    m[("erode", "r1", "scipy")] = lambda: scipy_erode(img, box(1))
    m[("erode", "r1", "scipy (binary)")] = lambda: ndi.binary_erosion(bin_img, box(1))

    m[("dilate", "r1", "nbmorph")] = lambda: nbmorph.dilate_labels_spherical(img, 1, struct_sequence="B")
    m[("dilate", "r1", "fastmorph")] = lambda: fastmorph.dilate(img, background_only=True, parallel=p)
    m[("dilate", "r1", "scipy")] = lambda: scipy_dilate(img, box(1))
    m[("dilate", "r1", "scipy (binary)")] = lambda: ndi.binary_dilation(bin_img, box(1))

    # ---- radius R, quasi-spherical / sphere / cube ----
    m[("erode", "rR", "nbmorph")] = lambda: nbmorph.erode_labels_spherical(img, radius)
    m[("erode", "rR", "fastmorph")] = lambda: fastmorph.erode(img, parallel=p, iterations=radius)
    m[("erode", "rR", "fastmorph (EDT)")] = lambda: fastmorph.spherical_erode(img, radius=radius, parallel=p)
    m[("erode", "rR", "scipy (box^R)")] = lambda: scipy_erode_iter(img, radius)
    m[("erode", "rR", "scipy")] = lambda: scipy_erode(img, fp_ball)
    m[("erode", "rR", "scipy (binary)")] = lambda: ndi.binary_erosion(bin_img, fp_ball)

    m[("dilate", "rR", "nbmorph")] = lambda: nbmorph.dilate_labels_spherical(img, radius)
    m[("dilate", "rR", "fastmorph")] = lambda: fastmorph.dilate(img, background_only=True, parallel=p, iterations=radius)
    m[("dilate", "rR", "scipy (box^R)")] = lambda: scipy_dilate_iter(img, radius)
    m[("dilate", "rR", "scipy")] = lambda: scipy_dilate(img, fp_ball)
    m[("dilate", "rR", "scipy (binary)")] = lambda: ndi.binary_dilation(bin_img, fp_ball)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--radius", type=int, default=3)
    ap.add_argument("--threads", type=int, nargs="+", default=[1, numba.config.NUMBA_NUM_THREADS])
    ap.add_argument("--data", default=str(ROOT / "data" / "dense_cells.npz"))
    ap.add_argument("--out", default=str(ROOT / "img" / "benchmark.png"))
    args = ap.parse_args()

    raw = np.load(args.data)["arr_0"]
    # The raw volume is almost gap-free; one erosion turns it into a realistic
    # segmentation with background between the cells, so dilation has work to do.
    img = np.ascontiguousarray(nbmorph.erode_labels_spherical(raw, 1))
    print(f"input {img.shape} {img.dtype}, {len(np.unique(img))} labels, "
          f"{100 * (img == 0).mean():.1f}% background, {img.nbytes / 1e6:.0f} MB")

    # effective structuring element per method (measured, not assumed)
    fp_sizes = {
        ("r1", "nbmorph"): effective_footprint(lambda a: nbmorph.dilate_labels_spherical(a, 1, struct_sequence="B"), 1),
        ("r1", "fastmorph"): effective_footprint(lambda a: fastmorph.dilate(a, parallel=1), 1),
        ("r1", "scipy"): int(box(1).sum()),
        ("r1", "scipy (binary)"): int(box(1).sum()),
        ("rR", "nbmorph"): effective_footprint(lambda a: nbmorph.dilate_labels_spherical(a, args.radius), args.radius),
        ("rR", "fastmorph"): int(box(args.radius).sum()),
        ("rR", "scipy (box^R)"): int(box(args.radius).sum()),
        ("rR", "fastmorph (EDT)"): int(ball(args.radius).sum()),
        ("rR", "scipy"): int(ball(args.radius).sum()),
        ("rR", "scipy (binary)"): int(ball(args.radius).sum()),
    }
    print("effective structuring elements (voxels):", fp_sizes)

    rows = []
    for threads in args.threads:
        numba.set_num_threads(threads)
        methods = build_methods(img, args.radius, threads)

        print(f"\n--- {threads} thread(s) ---")
        # warm-up (JIT compile, thread pool, page cache) + correctness snapshot
        results = {}
        for key, fn in methods.items():
            results[key] = fn()

        for (op, grp) in [("erode", "r1"), ("dilate", "r1"), ("erode", "rR"), ("dilate", "rR")]:
            ref = results[(op, grp, "nbmorph")]
            for key, fn in methods.items():
                if key[:2] != (op, grp):
                    continue
                lib = key[2]
                out = results[key]
                agr = "reference" if lib == "nbmorph" else agreement(
                    ref, out.astype(ref.dtype) if out.dtype == bool else out)
                tmin, tmed, n = timeit(fn, args.repeats)
                rows.append(dict(threads=threads, op=op, group=grp, lib=lib,
                                 t_min=tmin, t_med=tmed, reps=n,
                                 footprint=fp_sizes[(grp, lib)], agreement=agr))
                print(f"{op:6s} {grp:3s} {lib:18s} {tmin*1e3:9.1f} ms  "
                      f"(median {tmed*1e3:7.1f}, n={n})  fp={fp_sizes[(grp,lib)]:4d}  {agr}")

    csv_path = Path(args.out).with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {csv_path}")

    plot(rows, args, img)


def plot(rows, args, img):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    # validated categorical slots (blue / orange / aqua) + within-family tints
    C = {"nbmorph": "#2a78d6",
         "fastmorph": "#eb6834", "fastmorph (EDT)": "#f4a985",
         "scipy": "#1baf7a", "scipy (box^R)": "#5fcda3", "scipy (binary)": "#a8e2c9"}
    LABEL = {"scipy (box^R)": f"scipy, {args.radius}x box",
             "scipy (binary)": "scipy, binary only",
             "fastmorph (EDT)": "fastmorph, EDT sphere"}
    order = ["nbmorph", "fastmorph", "fastmorph (EDT)",
             "scipy", "scipy (box^R)", "scipy (binary)"]
    threads = sorted({r["threads"] for r in rows})
    t_lo, t_hi = threads[0], threads[-1]
    R = args.radius
    groups = [("r1", "radius 1  ·  3x3x3 box"), ("rR", f"radius {R}  ·  spherical")]
    ops = [("erode", "erosion"), ("dilate", "dilation")]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.4))
    xmax = max(r["t_min"] for r in rows) * 1e3

    for i, (grp, gname) in enumerate(groups):
        for j, (op, opname) in enumerate(ops):
            ax = axes[i][j]
            sel = [r for r in rows if r["op"] == op and r["group"] == grp]
            libs = sorted({r["lib"] for r in sel}, key=order.index)
            y = np.arange(len(libs))[::-1].astype(float)

            def get(lib, t):
                v = [r for r in sel if r["lib"] == lib and r["threads"] == t]
                return v[0] if v else None

            h = 0.36
            for k, lib in enumerate(libs):
                for t, off, alpha, hatch in ((t_lo, +h / 2, 0.45, "///"),
                                             (t_hi, -h / 2, 1.0, None)):
                    r = get(lib, t)
                    if r is None:
                        continue
                    v = r["t_min"] * 1e3
                    ax.barh(y[k] + off, v, height=h - 0.04, color=C[lib],
                            alpha=alpha, hatch=hatch,
                            edgecolor=C[lib] if hatch else "none", linewidth=0)
                    mark = "" if r["agreement"] in ("reference", "identical") else (
                        " ~" if r["agreement"] == "same support" else " *")
                    ax.text(v * 1.09, y[k] + off, f"{v:,.0f} ms{mark}",
                            va="center", fontsize=8.5, color="#52514e")

            ax.set_yticks(y)
            ax.set_yticklabels(
                [f"{LABEL.get(l, l)}\n{get(l, t_hi)['footprint']} voxels" for l in libs],
                fontsize=9)
            for tick, lib in zip(ax.get_yticklabels(), libs):
                if lib == "nbmorph":
                    tick.set_fontweight("bold")
            ax.set_xscale("log")
            ax.set_xlim(7, xmax * 3.2)
            ax.set_ylim(-0.7, len(libs) - 0.3)
            ax.grid(axis="x", ls=":", color="0.8")
            ax.set_axisbelow(True)
            for sp in ("top", "right", "left"):
                ax.spines[sp].set_visible(False)
            ax.tick_params(axis="y", length=0)
            ax.set_title(f"{opname}   —   {gname}", fontsize=11, loc="left",
                         color="#0b0b0b", pad=8)
            if i == 1:
                ax.set_xlabel("runtime  [ms, log scale]   ·   lower is better",
                              fontsize=9.5, color="#52514e")

    handles = [Patch(facecolor="#7d7d7d", alpha=0.45, hatch="///",
                     label=f"{t_lo} thread"),
               Patch(facecolor="#7d7d7d", label=f"{t_hi} threads  (scipy is single-threaded)")]
    fig.legend(handles=handles, fontsize=9.5, frameon=False, ncol=2,
               loc="upper right", bbox_to_anchor=(0.995, 0.995))

    fig.suptitle("Multi-label 3D morphology: nbmorph vs. fastmorph vs. scipy.ndimage",
                 fontsize=14, x=0.012, ha="left", y=0.985)
    fig.text(0.012, 0.935,
             f"{img.shape[0]}x{img.shape[1]}x{img.shape[2]} uint16 EM segmentation, "
             f"{len(np.unique(img))} labels, {100*(img==0).mean():.0f}% background "
             f"({img.nbytes/1e6:.0f} MB)  ·  Apple M4  ·  best of N runs, "
             f"compilation excluded  ·  structuring element size given per bar",
             fontsize=9, color="#52514e")
    fig.text(0.012, 0.030,
             "Output is bit-identical to nbmorph unless marked.   "
             "~ same voxels changed, different tie-break among equally frequent labels.",
             fontsize=8.5, color="#6b6a66")
    fig.text(0.012, 0.008,
             "* different structuring element or rule: scipy dilation assigns the "
             "neighbourhood max, not the mode; scipy binary drops the labels entirely.",
             fontsize=8.5, color="#6b6a66")
    fig.tight_layout(rect=[0, 0.055, 1, 0.915])
    fig.subplots_adjust(hspace=0.34, wspace=0.30)
    fig.savefig(args.out, dpi=200, facecolor="white")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
