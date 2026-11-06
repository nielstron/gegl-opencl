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
 *
 * The prefix sums of a whole row can be many orders of magnitude larger
 * than the sum of a single span (highlight weights reach 1024), so a span
 * computed as the difference of two float prefix sums would lose most of
 * its precision.  The prefix sums are therefore kept as unevaluated sums
 * of two floats (hi + lo, using error-free additions), which makes the
 * span sums about as accurate as the CPU sliding window.  This relies on
 * the additions not being reassociated, which GEGL ensures by building
 * the kernels with -cl-opt-disable; no fma or fp64 is needed.
 *
 * The scan size is the local work size, a power of two chosen by the
 * host from the kernel's work group size limit.
 */

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

/* (a_hi + a_lo) + (b_hi + b_lo), renormalized; Knuth's two-sum */
static void
dd_add4 (float4 *a_hi, float4 *a_lo, float4 b_hi, float4 b_lo)
{
  float4 s  = *a_hi + b_hi;
  float4 bb = s - *a_hi;
  float4 e  = (*a_hi - (s - bb)) + (b_hi - bb);
  float4 h;

  e += *a_lo + b_lo;

  h  = s + e;
  bb = h - s;
  *a_lo = (s - (h - bb)) + (e - bb);
  *a_hi = h;
}

static void
dd_add (float *a_hi, float *a_lo, float b_hi, float b_lo)
{
  float s  = *a_hi + b_hi;
  float bb = s - *a_hi;
  float e  = (*a_hi - (s - bb)) + (b_hi - bb);
  float h;

  e += *a_lo + b_lo;

  h  = s + e;
  bb = h - s;
  *a_lo = (s - (h - bb)) + (e - bb);
  *a_hi = h;
}

__kernel void
lens_blur_prefix (__global const float4 *src,
                  __global       float4 *prefix,     /* hi, lo pairs */
                  __global       float2 *prefix_w,   /* hi, lo */
                  __local        float4 *scan,       /* 2 * local size */
                  __local        float2 *scan_w,     /* local size */
                  int                    width,
                  int4                   clip,       /* x0, y0, x1, y1 */
                  float4                 luminance,
                  float                  threshold_low,
                  float                  threshold_high,
                  float                  factor,
                  float                  highlight_max)
{
  int    lid       = get_local_id (0);
  int    scan_size = get_local_size (0);
  int    y         = get_group_id (1);
  int    stride    = width + 1;
  float4 carry     = (float4) (0.0f);
  float4 carry_lo  = (float4) (0.0f);
  float  carry_w   = 0.0f;
  float  carry_wlo = 0.0f;
  int    base;

  if (lid == 0)
    {
      prefix[2 * y * stride]     = (float4) (0.0f);
      prefix[2 * y * stride + 1] = (float4) (0.0f);
      prefix_w[y * stride]       = (float2) (0.0f);
    }

  for (base = 0; base < width; base += scan_size)
    {
      int    x = base + lid;
      float4 v = (float4) (0.0f);
      float  w = 0.0f;
      float4 hi, lo;
      float  w_hi, w_lo;
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

      scan[2 * lid]     = v;
      scan[2 * lid + 1] = (float4) (0.0f);
      scan_w[lid]       = (float2) (w, 0.0f);

      barrier (CLK_LOCAL_MEM_FENCE);

      /* Hillis-Steele inclusive scan */
      for (offset = 1; offset < scan_size; offset *= 2)
        {
          hi   = scan[2 * lid];
          lo   = scan[2 * lid + 1];
          w_hi = scan_w[lid].x;
          w_lo = scan_w[lid].y;

          if (lid >= offset)
            {
              float2 t_w = scan_w[lid - offset];

              dd_add4 (&hi, &lo,
                       scan[2 * (lid - offset)],
                       scan[2 * (lid - offset) + 1]);
              dd_add (&w_hi, &w_lo, t_w.x, t_w.y);
            }

          barrier (CLK_LOCAL_MEM_FENCE);

          scan[2 * lid]     = hi;
          scan[2 * lid + 1] = lo;
          scan_w[lid]       = (float2) (w_hi, w_lo);

          barrier (CLK_LOCAL_MEM_FENCE);
        }

      hi   = carry;
      lo   = carry_lo;
      w_hi = carry_w;
      w_lo = carry_wlo;

      dd_add4 (&hi, &lo, scan[2 * lid], scan[2 * lid + 1]);
      dd_add (&w_hi, &w_lo, scan_w[lid].x, scan_w[lid].y);

      if (x < width)
        {
          prefix[2 * (y * stride + x + 1)]     = hi;
          prefix[2 * (y * stride + x + 1) + 1] = lo;
          prefix_w[y * stride + x + 1]         = (float2) (w_hi, w_lo);
        }

      dd_add4 (&carry, &carry_lo,
               scan[2 * (scan_size - 1)], scan[2 * (scan_size - 1) + 1]);
      dd_add (&carry_w, &carry_wlo,
              scan_w[scan_size - 1].x, scan_w[scan_size - 1].y);

      barrier (CLK_LOCAL_MEM_FENCE);
    }
}

__kernel void
lens_blur_disc (__global const float4 *prefix,      /* hi, lo pairs */
                __global const float2 *prefix_w,    /* hi, lo */
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
      int    s   = spans[abs (r)];
      int    row = (y + iradius + r) * stride;
      int    xl  = row + x + iradius - s;
      int    xr  = row + x + iradius + s + 1;
      float2 wl  = prefix_w[xl];
      float2 wr  = prefix_w[xr];

      /* the difference of the high parts is close to the span sum, so it
       * is rounded relative to the span sum, not to the prefix sums
       */
      sum   += (prefix[2 * xr]     - prefix[2 * xl]) +
               (prefix[2 * xr + 1] - prefix[2 * xl + 1]);
      sum_w += (wr.x - wl.x) + (wr.y - wl.y);
    }

  result.xyz = sum.xyz / sum.w;
  result.w   = sum.w / sum_w;

  dst[y * get_global_size (0) + x] = result;
}
