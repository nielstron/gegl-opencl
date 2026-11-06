# GEGL OpenCL fork

This is a **general OpenCL fork of [GEGL](https://gitlab.gnome.org/GNOME/gegl)**. It
fixes GEGL's existing OpenCL kernels where they disagree with the CPU path, and adds
OpenCL implementations for operations that had none. It is not tied to one
application. It is developed for, and tested against, the needs of the Android editor
[Brushkin](https://github.com/nielstron/brushkin).

> **Status: work in progress.** Nothing is merged into `main` yet: `main` is GNOME
> master plus this README. Every change lives on its own branch (see below), and the
> table shows where each one stands. A combined branch will follow once the open
> issues are fixed.

## How it is organised

- Base: GNOME master (`gitlab.gnome.org/GNOME/gegl`, `7f032dd`).
- **One branch per change**, so each can be reviewed, upstreamed or dropped on its own:
  - `fix/<op>-opencl`: an existing OpenCL path that produced wrong output, crashed or
    never ran.
  - `opencl/<op>`: a new OpenCL implementation.
  - `fix/<core-topic>`: a fix in GEGL's core that the OpenCL work depends on or
    uncovered (threading, sampler, iterator, colour conversion).
  - `perf/<topic>`: a speed change that does not change output.
- The CPU path is the reference. A branch changes CPU output only where the CPU result
  was itself wrong (e.g. it depended on thread count or chunk size); the table marks those.

## How equivalence is checked

Every branch is compared against the CPU path using one harness. It renders a
deterministic test pattern twice, once with `use-opencl` off and once with it on, opaque
and with varying alpha (error weighted by alpha). It reports which OpenCL kernels
actually launched, so a silent CPU fallback cannot pass as a GPU result. Formats tested
are `R'G'B'A u8` (what Brushkin uses), `RGBA float` and others, at sizes from 1×1 to
3000×2000 so that GEGL's chunk and tile boundaries are crossed.

**Target:** the GPU output must not differ from the CPU by more than 1/255 in u8,
or 0.5/255 in float.

Each branch was then checked again by an independent reviewer that tried to break it:
untested parameters, odd sizes, other formats, a code review for portability to
Mali/Adreno, and a check that CPU output is unchanged.

Testing so far is on an Apple M3 Max (OpenCL 1.2 on Metal). Mali/Adreno results are
still to come.

## Results

Verdict comes from the independent review: **pass**, **pass\*** (pass with concerns,
see notes) or **fail** (being fixed). Error is the max per-channel error on the 0–255
scale, for u8 opaque / u8 alpha-weighted / RGBA float, as measured by the reviewer
across all the settings it tried (including extremes). Speedups are **provisional**:
they were measured on a heavily loaded machine, partly on battery, and against a CPU
path that GEGL wrongly ran single-threaded (see `fix/opencl-threading-gate`). They will
be re-measured.

### Fixes to existing OpenCL paths

| Branch | Problem on master | Verdict | Max error u8 / alpha / float | Provisional speedup | Notes |
|---|---|---|---|---|---|
| `fix/gblur-1d-opencl-apple` | gaussian-blur / gblur-1d FIR aborted Apple's OpenCL compiler | pass\* | 1 / 0.97 / 0.001 | 3–27× | `cltostring.py` wrote "é" in a kernel comment as an invalid UTF-8 byte. FIR still ignores the mipmap level (pre-existing). |
| `fix/texturize-canvas-opencl` | Process crashed (SIGBUS) | pass\* | 0 / 0 / 0.0001 | ~1.8× | Read-only table was wrapped with `CL_MEM_USE_HOST_PTR`. |
| `fix/edge-sobel-opencl` | Both-axes mode never ran on the GPU; single-axis wrong at chunk borders (up to 48/255) | pass\* | 1 / 1 / 0.0001 | ~1.2× | The CPU path samples one pixel off; that is kept, and worth an upstream fix. |
| `fix/red-eye-removal-opencl` | Up to 95/255 at threshold boundaries | pass | 0 / 0 / 0.00001 | ~1.6× | Threshold now computed like the CPU's double. |
| `fix/motion-blur-circular-opencl` | Up to 15/255; CPU itself depended on chunk size | pass\* | 255 / 248 / 42 at extreme angles | 21–36× | ≤1 at normal settings. Large errors at angle=360 and other large angles; being fixed. **Changes CPU output** (CPU bugs fixed). |
| `fix/contrast-curve-opencl` | Up to 3/255 | pass\* | 100 / 62 / 2.9 worst case | ~1.0× | No kernel bug: the error comes from GEGL's GPU colour conversion (see `fix/cl-color-babl-parity`). Not faster. |
| `fix/noise-reduction-opencl` | OpenCL path compiled out | pass\* | 0 / 0 / 0 | 4–9× | **Changes CPU output**: the CPU result depended on thread count (up to 72/255). The algorithm is chaotic, so exact match on Mali/Adreno is unlikely. Optional. |
| `fix/c2g-opencl-parity-master` | c2g ignored enhance-shadows and mis-projected | pass\* | n/a (stochastic) / 0 float | n/a | The deterministic kernel test passes. Pre-existing sampling differences remain. |

### New OpenCL implementations

| Branch | Verdict | Max error u8 / alpha / float | Provisional speedup | Notes |
|---|---|---|---|---|
| `opencl/gblur-1d-iir` | pass\* | 1 / 1.28 / 0.06 | 1.5–2.1× | IIR gaussian; this is what gaussian-blur's auto mode uses. It is used by unsharp-mask, high-pass, bloom, softglow, cartoon, photocopy and shadows-highlights. Stacked on `fix/gblur-1d-opencl-apple`. |
| `opencl/mean-curvature-blur` | pass\* | 5 / 4.6 / 6.1 at extremes | 60–87× | ≤1 at defaults. A subnormal gradient gives NaN on devices that keep denormals; being fixed. |
| `opencl/motion-blur-zoom` | pass | 1 / 0.62 / 0 | 6–10× | |
| `opencl/median-blur` | pass\* | 0 / 0 / 0 | 8–14× | Fixed 16×16 work group; must respect the device limit. |
| `opencl/lens-blur` | **fail** | 76 / 2.6 / 25.6 | 3–6× | Float prefix sums lose precision; being fixed. |
| `opencl/noise-pick` | pass | 0 / 0 / 0 | 10–14× | |
| `opencl/noise-slur` | pass | 0 / 0 / 0 | 12–14× | |
| `opencl/noise-spread` | pass | 0 / 0 / 0 | 11–12× | |
| `opencl/noise-rgb` | pass\* | 1 / 1 / 0.00003 | 4–5× | Also speeds up the shared `random.cl` about 2× (same output). |
| `opencl/noise-cie-lch` | pass\* | 1 / 1 / 0.001 | ~3× | Exact gray match assumes babl was built with FMA contraction. |
| `opencl/stretch-contrast-hsv` | **fail** | 1 / 1 / 0.005 | ~27× | Wrong for near-black pixels and non-sRGB spaces; being fixed. |
| `opencl/color-enhance` | **fail** | 1 / 1 / 0.08 | ~5.5× | Wrong for grayscale input, and CPU fallback after a GPU failure is corrupted; being fixed. |
| `opencl/normal-map` | pass\* | 1 / 1 / 0.01 | ~11× | NaN with premultiplied-float input that has fully transparent pixels. |
| `opencl/emboss` | **fail** | 1 / 1 / 0.009 | 7–9× | Can abort (format mismatch on fallback); being fixed. |
| `opencl/newsprint` | pass\* | 255 in edge cases | 5–16× | Bit-exact at normal settings. Exactness depends on CPU FMA contraction and libm `hypotf`; width or blocksize 0 differ. |
| `opencl/deinterlace` | pass\* | 1 / 1 / 0.001 | ~5× | Edge cases with size ≥ width differ; the CPU itself reads out of bounds there. |
| `opencl/whirl-pinch` | pass\* | 22 / 8 / 0.06 at extremes | ~13×+ | Ported NoHalo/cubic/linear samplers. Re-uploads the whole input per chunk; float-vs-double mapping at extreme settings. |
| `opencl/polar-coordinates` | pass\* | 1 / 1.8 / 0.07 | ~24× | `fmod` sign and a NaN at a pole on the top edge; being fixed. |
| `opencl/ripple` | **fail** | 255 / 13 / 0.1 | ~8× | Period passed as float (phase drift), and non-static `inline` helpers fail on some compilers; being fixed. |
| `opencl/waves` | **fail** | 5 / 3.7 / 1.1 | ~6× | Parameters passed as float instead of double; being fixed. |
| `opencl/lens-distortion` | pass\* | 28 / 75 / 23 at extremes | ~1.5× | Barely faster; may be dropped. |

### Core fixes

| Branch | What | Verdict | Notes |
|---|---|---|---|
| `fix/opencl-threading-gate` | GEGL turned off CPU threading for every OpenCL-capable op once OpenCL was initialised, even on nodes with `use-opencl=false` | pass | CPU path of bilateral-filter with 8 threads: 3.4–5.1 s → 0.85 s. Matters for any app that enables OpenCL per graph. |
| `fix/sampler-context-rect` | Sampler cache could miss the NoHalo context, so CPU output depended on thread count | pass | **Changes CPU output** (only where it was undefined). |
| `fix/cl-iterator-abyss-alpha` | Area filters on sources without alpha got opaque black instead of transparent at edges on the GPU (up to 150/255) | under review | Upstream's existing guard checked the wrong rectangle. |
| `fix/cl-color-babl-parity` | GPU colour conversions differ from babl in the last bits, flipping threshold/table ops | under review | |
| `perf/opencl-transfer` | Speed up GEGL's shared OpenCL transfer/conversion path | in progress | Simple point ops are transfer-bound today. |

### Not ported

mosaic, cubism, distance-transform, dither, lens-flare and rgb-clip are inherently
sequential, order-dependent, or too cheap to gain from the GPU.

## Building

Same as GEGL:

```sh
meson setup _build && ninja -C _build
```

Requires babl ≥ 0.1.116.
