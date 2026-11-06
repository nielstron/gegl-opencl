# GEGL OpenCL fork

This is a **general OpenCL fork of [GEGL](https://gitlab.gnome.org/GNOME/gegl)**.
It fixes GEGL's OpenCL kernels where they disagree with the CPU path, crash or
never run. It adds OpenCL implementations for operations that had none. It also
removes most of the overhead GEGL's OpenCL path spends around the kernels.

The fork is not tied to one application. It was developed against the needs of
the Android editor [Brushkin](https://github.com/nielstron/brushkin) and is
tested on Android hardware.

## Branches

`main` is GNOME master (`gitlab.gnome.org/GNOME/gegl`, `f51b97273`) with every
change below merged in. Each change also lives on its own branch, so it can be
reviewed, upstreamed or dropped by itself:

- `fix/<op>-opencl`: an existing OpenCL path that gave wrong output, crashed or
  never ran.
- `opencl/<op>`: a new OpenCL implementation.
- `perf/<topic>`: a speedup that does not change output.
- `fix/<topic>`: a fix in GEGL's core (threading, sampler, iterator, colour
  conversion) that the OpenCL work depends on or uncovered.

Most branches sit directly on GNOME master. These sit on other branches:

- `opencl/gblur-1d-iir` sits on `fix/gblur-1d-opencl-apple`.
- `fix/motion-blur-circular-opencl` sits on `fix/cl-iterator-abyss-alpha`.
- `perf/edge-sobel-opencl` sits on `fix/edge-sobel-opencl`.
- `perf/high-pass-opencl` sits on a merge of `perf/opencl-transfer` and
  `opencl/gblur-1d-iir`.

`fix/contrast-curve-opencl` is published but not merged. Its kernel was already
correct, and it is not faster.

## How equivalence is checked

The CPU path is the reference. A branch changes CPU output only where the CPU
result was itself wrong. Each such case is a separate commit, listed under
"CPU behaviour changes" below.

1. **Harness, on desktop:** every branch was compared against the CPU on an
   Apple M3 Max. Each filter was rendered with `use-opencl` off and on, opaque
   and with varying alpha. The tests used several formats (`R'G'B'A u8`,
   `RGBA float`, others) and sizes from 1×1 to 4000×3000, so they cross GEGL's
   chunk and tile boundaries. The harness records which OpenCL kernels actually
   launched, so a silent CPU fallback cannot pass as a GPU result.
2. **Independent review:** a separate reviewer then tried to break each branch
   with untested parameters, odd sizes, other formats, fault injection (forced
   GPU failures), in-place blits and Mali/Adreno portability checks. Branches
   went through fix-and-review rounds until they passed.
3. **On the target device:** Brushkin's OpenCL survey ran on a **Pixel 6 Pro
   (Mali-G78, Android 17)**, built against `main`.

Target: the GPU output is within 1/255 of the CPU, opaque and alpha-weighted.

## Results on a Pixel 6 Pro

All tests ran at 1024×768, `R'G'B'A u8` (Brushkin's format). Each value is the
median of 3 interleaved rounds.

- **CPU:** GEGL's CPU path with 2 threads, as Brushkin runs it.
- **Max error:** per channel on the 0–255 scale. The alpha value is
  alpha-weighted. c2g samples randomly, so it never matches exactly.

GPU time includes a fixed cost of about 60 ms per render on this device. It
covers uploading, converting and downloading the image, plus the app's own
pixel copies. Cheap per-pixel filters therefore top out near 2× at this size,
and larger images gain more.

| Filter | Kind | CPU (2 threads) | GPU | Speedup | Max error opaque / alpha |
|---|---|---|---|---|---|
| `diffraction-patterns` | upstream | 3412 ms | 91 ms | 37.4× | 1 / 1.0 |
| `c2g` | fixed | 3933 ms | 145 ms | 27.0× | random |
| `motion-blur-circular` | fixed | 1078 ms | 87 ms | 12.3× | 1 / 0.9 |
| `mean-curvature-blur` | new | 1025 ms | 87 ms | 11.8× | 0 / 1.0 |
| `bilateral-filter` | upstream | 763 ms | 72 ms | 10.6× | 1 / 0.9 |
| `stretch-contrast-hsv` | new | 536 ms | 67 ms | 8.0× | 1 / 1.0 |
| `ripple` | new | 489 ms | 70 ms | 6.9× | 1 / 1.9 |
| `polar-coordinates` | new | 1061 ms | 155 ms | 6.8× | 1 / 1.0 |
| `gaussian-blur-selective` | upstream | 482 ms | 73 ms | 6.6× | 0 / 0.0 |
| `cartoon` | new | 506 ms | 79 ms | 6.4× | 1 / 1.0 |
| `motion-blur-linear` | upstream | 426 ms | 69 ms | 6.2× | 1 / 0.8 |
| `waves` | new | 435 ms | 74 ms | 5.9× | 1 / 1.0 |
| `high-pass` | new | 479 ms | 92 ms | 5.2× | 1 / 1.0 |
| `noise-hsv` | faster | 433 ms | 83 ms | 5.2× | 1 / 1.0 |
| `noise-reduction` | enabled | 428 ms | 94 ms | 4.5× | 0 / 0.0 |
| `normal-map` | new | 277 ms | 62 ms | 4.5× | 1 / 1.0 |
| `snn-mean` | upstream | 294 ms | 67 ms | 4.4× | 0 / 0.0 |
| `motion-blur-zoom` | new | 321 ms | 75 ms | 4.3× | 1 / 0.0 |
| `noise-pick` | new | 206 ms | 62 ms | 3.3× | 0 / 0.0 |
| `noise-cie-lch` | new | 255 ms | 79 ms | 3.2× | 1 / 1.0 |
| `color-enhance` | new | 213 ms | 67 ms | 3.2× | 1 / 1.0 |
| `noise-slur` | new | 202 ms | 64 ms | 3.2× | 0 / 0.0 |
| `noise-spread` | new | 215 ms | 68 ms | 3.2× | 0 / 0.0 |
| `lens-blur` | new | 275 ms | 89 ms | 3.1× | 0 / 0.5 |
| `noise-rgb` | new | 271 ms | 89 ms | 3.0× | 1 / 1.0 |
| `photocopy` | new | 255 ms | 84 ms | 3.0× | 1 / 1.0 |
| `posterize` | upstream | 187 ms | 62 ms | 3.0× | 0 / 0.0 |
| `oilify` | upstream | 1512 ms | 526 ms | 2.9× | 1 / 1.0 |
| `softglow` | new | 225 ms | 79 ms | 2.8× | 1 / 1.0 |
| `gaussian-blur#fir` | fixed (Apple) | 234 ms | 82 ms | 2.8× | 0 / 0.3 |
| `newsprint` | new | 179 ms | 64 ms | 2.8× | 0 / 0.0 |
| `gblur-1d#fir` | fixed (Apple) | 174 ms | 66 ms | 2.6× | 0 / 0.9 |
| `emboss` | new | 159 ms | 61 ms | 2.6× | 1 / 1.0 |
| `edge-sobel#abs` | enabled | 157 ms | 61 ms | 2.6× | 0 / 0.0 |
| `edge-sobel#both` | enabled | 156 ms | 62 ms | 2.5× | 0 / 0.0 |
| `lens-distortion` | new | 183 ms | 74 ms | 2.5× | 1 / 1.0 |
| `bloom` | new | 219 ms | 89 ms | 2.5× | 1 / 1.0 |
| `edge-sobel` | faster | 150 ms | 62 ms | 2.4× | 0 / 0.0 |
| `brightness-contrast` | upstream | 148 ms | 63 ms | 2.4× | 0 / 0.0 |
| `box-blur` | faster | 161 ms | 69 ms | 2.3× | 1 / 0.8 |
| `color-temperature` | upstream | 141 ms | 62 ms | 2.3× | 0 / 0.0 |
| `whirl-pinch` | new | 898 ms | 399 ms | 2.3× | 1 / 1.0 |
| `pixelize` | fixed, faster | 142 ms | 63 ms | 2.2× | 0 / 0.0 |
| `invert-linear` | upstream | 138 ms | 62 ms | 2.2× | 0 / 0.0 |
| `gaussian-blur#small-auto` | upstream | 174 ms | 78 ms | 2.2× | 0 / 0.0 |
| `levels` | upstream | 133 ms | 62 ms | 2.1× | 0 / 0.0 |
| `gblur-1d#auto` | enabled (IIR) | 161 ms | 76 ms | 2.1× | 0 / 0.8 |
| `deinterlace` | new | 135 ms | 63 ms | 2.1× | 0 / 1.0 |
| `median-blur` | new | 221 ms | 104 ms | 2.1× | 0 / 0.0 |
| `mono-mixer` | upstream | 132 ms | 62 ms | 2.1× | 1 / 0.9 |
| `gray` | upstream | 132 ms | 66 ms | 2.0× | 0 / 0.0 |
| `vignette` | upstream | 124 ms | 62 ms | 2.0× | 1 / 1.0 |
| `contrast-curve` | upstream | 121 ms | 62 ms | 2.0× | 0 / 0.0 |
| `stretch-contrast` | upstream | 130 ms | 67 ms | 2.0× | 0 / 0.0 |
| `gaussian-blur#auto` | enabled (IIR) | 179 ms | 92 ms | 1.9× | 0 / 0.8 |
| `exposure` | upstream | 117 ms | 62 ms | 1.9× | 0 / 0.0 |
| `checkerboard` | upstream | 106 ms | 60 ms | 1.8× | 0 / 0.0 |
| `weighted-blend` | upstream | 168 ms | 98 ms | 1.7× | 0 / 0.0 |
| `svg:src-over` | upstream | 163 ms | 98 ms | 1.7× | 0 / 0.0 |
| `simplex-noise` | upstream | 125 ms | 76 ms | 1.6× | 1 / 1.0 |
| `edge-laplace` | upstream | 96 ms | 63 ms | 1.5× | 0 / 0.0 |
| `video-degradation` | upstream | 96 ms | 65 ms | 1.5× | 0 / 0.0 |
| `cell-noise` | upstream | 132 ms | 89 ms | 1.5× | 1 / 1.0 |
| `texturize-canvas` | fixed | 93 ms | 63 ms | 1.5× | 0 / 0.0 |
| `alien-map` | upstream | 86 ms | 63 ms | 1.4× | 0 / 0.0 |
| `opacity` | upstream | 78 ms | 63 ms | 1.2× | 0 / 0.0 |
| `noise-hurl` | upstream | 90 ms | 76 ms | 1.2× | 0 / 0.0 |
| `channel-mixer` | upstream | 157 ms | 135 ms | 1.2× | 0 / 0.0 |
| `value-invert` | faster | 69 ms | 60 ms | 1.1× | 1 / 1.0 |
| `threshold` | upstream | 215 ms | 191 ms | 1.1× | 0 / 0.0 |
| `color-exchange` | upstream | 68 ms | 63 ms | 1.1× | 0 / 0.0 |
| `red-eye-removal` | fixed | 67 ms | 62 ms | 1.1× | 0 / 0.0 |
| `shadows-highlights` | new | 212 ms | 202 ms | 1.0× | 1 / 1.0 |
| `unsharp-mask` | new | 190 ms | 195 ms | 1.0× | 0 / 0.0 |
| `shadows-highlights-correction` | upstream | 183 ms | 198 ms | 0.9× | 1 / 1.0 |
| `hue-chroma` | upstream | 156 ms | 173 ms | 0.9× | 0 / 0.0 |
| `color-to-alpha` | upstream | 75 ms | 106 ms | 0.7× | 0 / 0.0 |

## Core changes

| Branch | What |
|---|---|
| `perf/opencl-transfer` | Removes most of the overhead around OpenCL kernels. 8-bit conversions now run on the GPU and match babl exactly, via host-built tables. Device buffers are pooled. A default `gegl_node_blit` renders straight into the caller's memory. If the destination aliases a source buffer, it renders into a temporary instead, and if any GPU step fails it falls back to the CPU. It also fixes the iterator's converting write path, which wrote the wrong format. On the Pixel, GPU time for the filters Brushkin used went from about 130 ms to 62–71 ms. |
| `fix/opencl-threading-gate` | GEGL disabled CPU threading for every OpenCL-capable operation once OpenCL was initialised, even on nodes with `use-opencl=false`. Now only nodes that actually use OpenCL are serialised. |
| `fix/cl-color-babl-parity` | GEGL's GPU colour conversions now match babl bit-exactly on normal inputs. Before, the last-bit differences flipped threshold- and table-based filters by up to 255. The tables are built lazily, so startup memory stays at master's level. |
| `fix/cl-iterator-abyss-alpha` | Area filters on sources without alpha got opaque black instead of transparent at image edges on the GPU, by up to 150/255. Upstream's guard checked the wrong rectangle. |
| `fix/sampler-context-rect` | The sampler cache could miss NoHalo's context, which made CPU output depend on thread count. |
| `fix/opacity-opencl-components` | gegl:opacity's kernels assumed 4 channels and produced garbage on Y/YA input. |
| `fix/android-opencl-library` | Loads OpenCL on Android (`libOpenCL.so`, `libOpenCL-pixel.so`). |
| `fix/buffer-write-empty-span` | Skips empty spans when converting in `gegl_buffer_iterate_write`. |

## CPU behaviour changes

These commits change CPU output where the CPU result was itself wrong:

- **`fix/noise-reduction-opencl`:** the in-place algorithm depended on thread
  count (GNOME issue #162). It is now deterministic and matches the kernel.
- **`fix/motion-blur-circular-opencl`:** output depended on chunk size, and
  samples near powers of two read the wrong pixel.
- **`fix/sampler-context-rect`:** see above.
- **`opencl/color-enhance`:** achromatic input is now left unchanged instead of
  amplifying rounding noise.
- **`fix/lens-blur-cpu-precision`:** accumulates in double precision.

## Known limits

- **FMA dependence:** some kernels match the CPU exactly only when the CPU code
  was built with fused multiply-add contraction, as clang on arm64 does. That
  includes the colour-conversion paths and newsprint. On armv7 or x86 without
  FMA, rare boundary pixels can differ.
- **polar-coordinates:** one unexplained single-pixel difference appeared under
  heavy concurrent load on the desktop. It did not reproduce in repeated runs.
- **Concurrency:** blitting OpenCL graphs from several threads at once is racy,
  as it is upstream, because kernels are shared. Serialise renders.
- **Not ported:** mosaic, cubism, distance-transform, dither, lens-flare and
  rgb-clip are sequential, order-dependent or too cheap to gain from the GPU.

## Building

Same as GEGL:

```sh
meson setup _build && ninja -C _build
```

Requires babl ≥ 0.1.116; the tests used 0.1.128.
