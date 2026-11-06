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

/* sRGB transfer function, see babl/base/util.h */
static float
linear_to_gamma (float value)
{
  if (value > 0.003130804954f)
    return 1.055f * powr (value, 1.0f / 2.4f) - 0.055f;
  return 12.92f * value;
}

/* The work is expressed along the deinterlaced axis ("lines"): for the
 * horizontal orientation pos is the row and line_step the input row
 * stride, for the vertical one pos is the column and line_step 1. All
 * line numbers are absolute and are mapped to the input texture, which
 * starts at in_origin and has a border of size + 1 lines, with
 * in_offset. The line selection reproduces the CPU code, including the
 * use of the first line (inter) at the top / left edge, and, for the
 * vertical orientation, columns past the right edge of the CPU's source
 * rectangle (rect_x, rect_width) wrapping around to the start of the next
 * row, which the host provides in wrap (size columns from rect_x, for the
 * rows from wrap_y on).
 */
static float4
fetch_column (__global const float4 *in,
              __global const float4 *wrap,
                             int     x,
                             int     y,
                             int     size,
                             int     in_width,
                             int     in_x,
                             int     in_y,
                             int     rect_x,
                             int     rect_width,
                             int     wrap_y)
{
  int col = x - rect_x;

  if (col >= rect_width)
    return wrap[(y + 1 - wrap_y) * size + col - rect_width];

  return in[(y - in_y) * in_width + x - in_x];
}

__kernel void
deinterlace (__global const float4 *in,
             __global const float4 *wrap,
             __global       float4 *out,
                            int     horizontal,
                            int     keep,
                            int     size,
                            int     roi_x,
                            int     roi_y,
                            int     in_width,
                            int     in_x,
                            int     in_y,
                            int     extent,
                            int     rect_origin,
                            int     rect_x,
                            int     rect_width,
                            int     wrap_y,
                            int     to_gamma)
{
  int    gx     = get_global_id (0);
  int    gy     = get_global_id (1);
  int    width  = get_global_size (0);
  int    x      = roi_x + gx;
  int    y      = roi_y + gy;
  int    pos    = horizontal ? y : x;
  int    ix     = x - in_x;
  int    iy     = y - in_y;
  float4 center = in[iy * in_width + ix];
  float4 out_v  = center;

  if ((keep == 0 && pos % 2 == 0) || (keep == 1 && pos % 2 != 0))
    {
      float3 acc   = (float3) (0.0f);
      float  alpha = 0.0f;
      float  a;

      for (int i = 0; i < size; i++)
        {
          int    up, low;
          float4 u, l;

          up = (pos - i > 0) ? pos - i : rect_origin + keep;

          if (pos + i + 1 < extent)
            low = pos + i + 1;
          else if (horizontal)
            low = pos - 1 + keep;
          else
            low = pos + i - 1 + keep;

          if (horizontal)
            {
              u = in[(up  - in_y) * in_width + ix];
              l = in[(low - in_y) * in_width + ix];
            }
          else
            {
              u = fetch_column (in, wrap, up, y, size, in_width, in_x, in_y,
                                rect_x, rect_width, wrap_y);
              l = fetch_column (in, wrap, low, y, size, in_width, in_x, in_y,
                                rect_x, rect_width, wrap_y);
            }

          alpha += u.w + l.w;
          acc   += u.xyz * u.w + l.xyz * l.w;
        }

      a = alpha / (2 * size);
      out_v.w = a;

      if (a != 0.0f)
        out_v.xyz = acc / alpha;
    }

  if (to_gamma)
    out_v.xyz = (float3) (linear_to_gamma (out_v.x),
                          linear_to_gamma (out_v.y),
                          linear_to_gamma (out_v.z));

  out[gy * width + gx] = out_v;
}
