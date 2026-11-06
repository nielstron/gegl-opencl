/* This file is an image processing operation for GEGL
 *
 * GEGL is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * GEGL is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GEGL; if not, see <https://www.gnu.org/licenses/>.
 */

#define THRESHOLD   1.0f
#define N_BINS      100
#define ALPHA_FLOOR (1.0f / 65536.0f)

/* The images are kept as (Y * A, A): the YaA float pixels that
 * gegl:gaussian-blur works on after gegl:grey in cartoon.c.
 */

/* YaA float -> Y' float, like babl */
static float
grey (const float2 v)
{
  const float a = (v.y > ALPHA_FLOOR || v.y < -ALPHA_FLOOR) ? v.y : ALPHA_FLOOR;
  const float y = v.x / a;

  if (y > 0.003130804954f)
    return 1.055f * pow (y, 1.0f / 2.4f) - 0.055f;

  return 12.92f * y;
}

/* Histogram of grey1 / grey2 in [0, 1) over the n_pixels pixels of the
 * images, in 1 / 100 steps like compute_ramp ().
 */
__kernel void cartoon_hist (const global float2 *blur1,
                            const global float2 *blur2,
                                  global int    *hist,
                            const        int     n_pixels)
{
  local int lhist[N_BINS];
  int       i;

  for (i = get_local_id (0); i < N_BINS; i += get_local_size (0))
    lhist[i] = 0;

  barrier (CLK_LOCAL_MEM_FENCE);

  for (i = get_global_id (0); i < n_pixels; i += get_global_size (0))
    {
      const float grey1 = grey (blur1[i]);
      const float grey2 = grey (blur2[i]);

      if (grey2 != 0.0f)
        {
          /* diff = grey1 / grey2 in [0, 1), decided without rounding */
          const int in_range = grey2 > 0.0f ? (grey1 >= 0.0f && grey1 < grey2)
                                            : (grey1 <= 0.0f && grey1 > grey2);

          if (in_range)
            {
              const float diff = grey1 / grey2;
              int         bin  = (int) (diff * 100.0f);

              if (fma (diff, 100.0f, -(float) bin) < 0.0f)
                bin--;

              atomic_inc (&lhist[clamp (bin, 0, N_BINS - 1)]);
            }
        }
    }

  barrier (CLK_LOCAL_MEM_FENCE);

  for (i = get_local_id (0); i < N_BINS; i += get_local_size (0))
    if (lhist[i])
      atomic_add (&hist[i], lhist[i]);
}

/* Processes a chunk of R'G'B'A float input pixels into R'G'B'A float
 * output pixels; (x0, y0) is the chunk's position in the blurred images,
 * which are stride pixels wide.
 */
__kernel void cartoon_process (const global float4 *in,
                                     global float4 *out,
                               const global float2 *blur1,
                               const global float2 *blur2,
                               const        int     stride,
                               const        int     x0,
                               const        int     y0,
                               const        int     width,
                               const        float   ramp,
                               const        int     ramp_is_zero)
{
  const int    gid   = get_global_id (0);
  const int    idx   = (y0 + gid / width) * stride + x0 + gid % width;
  const float  grey1 = grey (blur1[idx]);
  const float  grey2 = grey (blur2[idx]);
  const float4 v     = in[gid];
  float        mult  = 0.0f;
  float        y, cb, cr;

  if (grey2 != 0.0f)
    {
      const float diff = grey1 / grey2;

      if (diff < THRESHOLD)
        {
          if (ramp_is_zero)
            {
              mult = 0.0f;
            }
          else
            {
              const float d = THRESHOLD - diff;

              mult = (ramp - (ramp < d ? ramp : d)) / ramp;
            }
        }
      else
        {
          mult = 1.0f;
        }
    }

  /* Y'CbCrA float as in babl, with the Y' channel replaced */
  y  = clamp (grey1 * mult, 0.0f, 1.0f);
  cb = -0.168736f * v.x - 0.331264f   * v.y + 0.5f      * v.z;
  cr =  0.5f      * v.x - 0.418688f   * v.y - 0.081312f * v.z;

  out[gid] = (float4) (y + 1.40200f * cr,
                       y - 0.344136f * cb - 0.71414136f * cr,
                       y + 1.772f * cb,
                       v.w);
}
