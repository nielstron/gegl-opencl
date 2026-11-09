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

#define THRESHOLD   0.75f
#define N_BINS      2000
#define ALPHA_FLOOR (1.0f / 65536.0f)

/* The images are kept as (Y * A, A), the luminance of the RaGaBaA float
 * pixels gegl:gaussian-blur works on in photocopy.c.
 */

/* RaGaBaA float -> Y float, like babl */
static float
luminance (const float2 v)
{
  const float a = (v.y > ALPHA_FLOOR || v.y < -ALPHA_FLOOR) ? v.y : ALPHA_FLOOR;

  return v.x * (1.0f / a);
}

/* Histogram of blur1 / blur2 below THRESHOLD over the width x height
 * region at (x0, y0) of images that are stride pixels wide, in 1 / 1000
 * steps like compute_ramp ().
 */
__kernel void photocopy_hist (const global float2 *blur1,
                              const global float2 *blur2,
                                    global int    *hist,
                              const        int     stride,
                              const        int     x0,
                              const        int     y0,
                              const        int     width,
                              const        int     n_pixels)
{
  local int lhist[N_BINS];
  int       i;

  for (i = get_local_id (0); i < N_BINS; i += get_local_size (0))
    lhist[i] = 0;

  barrier (CLK_LOCAL_MEM_FENCE);

  for (i = get_global_id (0); i < n_pixels; i += get_global_size (0))
    {
      const int   idx  = (y0 + i / width) * stride + x0 + i % width;
      const float diff = luminance (blur1[idx]) / luminance (blur2[idx]);

      if (diff < THRESHOLD && diff >= 0.0f)
        {
          /* (int) (diff * 1000) evaluated exactly */
          int bin = (int) (diff * 1000.0f);

          if (fma (diff, 1000.0f, -(float) bin) < 0.0f)
            bin--;

          atomic_inc (&lhist[clamp (bin, 0, N_BINS - 1)]);
        }
    }

  barrier (CLK_LOCAL_MEM_FENCE);

  for (i = get_local_id (0); i < N_BINS; i += get_local_size (0))
    if (lhist[i])
      atomic_add (&hist[i], lhist[i]);
}

/* Writes the width x height chunk at (x0, y0) of images that are stride
 * pixels wide, as Y float (n_comp = 1) or YA float (n_comp = 2).
 */
__kernel void photocopy_process (const global float2 *blur1,
                                 const global float2 *blur2,
                                       global float  *out,
                                 const        int     stride,
                                 const        int     x0,
                                 const        int     y0,
                                 const        int     width,
                                 const        int     n_comp,
                                 const        float   ramp_down,
                                 const        float   ramp_up)
{
  const int   gid  = get_global_id (0);
  const int   idx  = (y0 + gid / width) * stride + x0 + gid % width;
  const float y1   = luminance (blur1[idx]);
  const float diff = y1 / luminance (blur2[idx]);
  float       mult;
  float       value;

  /* MIN () as in photocopy.c, including NaN handling */
  if (diff < THRESHOLD)
    {
      if (ramp_down == 0.0f)
        {
          value = 0.0f;
        }
      else
        {
          const float d = THRESHOLD - diff;

          mult  = (ramp_down - (ramp_down < d ? ramp_down : d)) / ramp_down;
          value = y1 * mult;
        }
    }
  else
    {
      if (ramp_up == 0.0f)
        {
          mult = 1.0f;
        }
      else
        {
          const float d = diff - THRESHOLD;

          mult = (ramp_up < d ? ramp_up : d) / ramp_up;
        }

      value = mult + y1 - mult * y1;
    }

  if (n_comp == 1)
    {
      out[gid] = value;
    }
  else
    {
      out[2 * gid]     = value;
      out[2 * gid + 1] = 1.0f;
    }
}
