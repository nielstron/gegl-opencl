/* This file is an image processing operation for GEGL
 *
 * GEGL is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * GEGL is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GEGL; if not, see <https://www.gnu.org/licenses/>.
 */

/* Lens blur without a mask, in two passes:
 *
 *  1. lens_blur_prefix: one work group per source row computes the
 *     weighted, premultiplied pixels (r * a * w, g * a * w, b * a * w,
 *     a * w) and the weights w, and stores their exclusive prefix sums
 *     along the row (row length width + 1, starting with 0).  Pixels
 *     outside the clip rectangle are zeroed, which is equivalent to the
 *     CPU code clipping the neighborhood.
 *
 *  2. lens_blur_disc: every output pixel sums one span per disc row,
 *     using two prefix sum lookups per row.
 */

#define SCAN_SIZE 256

static float
highlight_weight (float4 pixel,
                  float4 luminance,
                  float  threshold_low,
                  float  threshold_high,
                  float  factor,
                  float  highlight_max)
{
  float v = dot (pixel.xyz, luminance.xyz);

  v = (v - threshold_low) / (threshold_high - threshold_low);

  if (v <= 0.0f)
    return 1.0f;
  else if (v >= 1.0f)
    return highlight_max;

  return exp (v * factor);
}

__kernel void
lens_blur_prefix (__global const float4 *src,
                  __global       float4 *prefix,
                  __global       float  *prefix_w,
                  __local        float4 *scan,
                  __local        float  *scan_w,
                  int                    width,
                  int4                   clip,       /* x0, y0, x1, y1 */
                  float4                 luminance,
                  float                  threshold_low,
                  float                  threshold_high,
                  float                  factor,
                  float                  highlight_max)
{
  int    lid     = get_local_id (0);
  int    y       = get_group_id (1);
  int    stride  = width + 1;
  float4 carry   = (float4) (0.0f);
  float  carry_w = 0.0f;
  int    base;

  if (lid == 0)
    {
      prefix[y * stride]   = (float4) (0.0f);
      prefix_w[y * stride] = 0.0f;
    }

  for (base = 0; base < width; base += SCAN_SIZE)
    {
      int    x = base + lid;
      float4 v = (float4) (0.0f);
      float  w = 0.0f;
      int    offset;

      if (x < width && x >= clip.s0 && x < clip.s2 && y >= clip.s1 && y < clip.s3)
        {
          float4 p = src[y * width + x];

          w = factor != 0.0f ?
                highlight_weight (p, luminance,
                                  threshold_low, threshold_high,
                                  factor, highlight_max) :
                1.0f;

          v.w   = p.w * w;
          v.xyz = p.xyz * v.w;
        }

      scan[lid]   = v;
      scan_w[lid] = w;

      barrier (CLK_LOCAL_MEM_FENCE);

      /* Hillis-Steele inclusive scan */
      for (offset = 1; offset < SCAN_SIZE; offset *= 2)
        {
          float4 t   = (float4) (0.0f);
          float  t_w = 0.0f;

          if (lid >= offset)
            {
              t   = scan[lid - offset];
              t_w = scan_w[lid - offset];
            }

          barrier (CLK_LOCAL_MEM_FENCE);

          scan[lid]   += t;
          scan_w[lid] += t_w;

          barrier (CLK_LOCAL_MEM_FENCE);
        }

      if (x < width)
        {
          prefix[y * stride + x + 1]   = carry   + scan[lid];
          prefix_w[y * stride + x + 1] = carry_w + scan_w[lid];
        }

      carry   += scan[SCAN_SIZE - 1];
      carry_w += scan_w[SCAN_SIZE - 1];

      barrier (CLK_LOCAL_MEM_FENCE);
    }
}

__kernel void
lens_blur_disc (__global const float4 *prefix,
                __global const float  *prefix_w,
                __global       float4 *dst,
                __constant     int    *spans,
                int                    src_width,
                int                    iradius)
{
  int    x      = get_global_id (0);
  int    y      = get_global_id (1);
  int    stride = src_width + 1;
  float4 sum    = (float4) (0.0f);
  float  sum_w  = 0.0f;
  int    r;
  float4 result;

  for (r = -iradius; r <= iradius; r++)
    {
      int s   = spans[abs (r)];
      int row = (y + iradius + r) * stride;
      int xl  = row + x + iradius - s;
      int xr  = row + x + iradius + s + 1;

      sum   += prefix[xr]   - prefix[xl];
      sum_w += prefix_w[xr] - prefix_w[xl];
    }

  result.xyz = sum.xyz / sum.w;
  result.w   = sum.w / sum_w;

  dst[y * get_global_size (0) + x] = result;
}
