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

#pragma OPENCL FP_CONTRACT OFF

/* Pack the pixels by column. Consecutive work items in the disc pass
 * handle consecutive rows and can read from a column together.
 */
static void
prepare_pixel (__global const float4 *src,
               __global const float  *mask,
               __global       float4 *pixels,
               __global       float2 *disc,
               float                  radius,
               float4                 luminance,
               float                  threshold_low,
               float                  threshold_high,
               float                  factor,
               float                  highlight_max,
               volatile __global int *fallback)
{
  int    x      = get_global_id (0);
  int    y      = get_global_id (1);
  int    index  = y * get_global_size (0) + x;
  int    target = x * get_global_size (1) + y;
  float4 p      = src[index];
  uint   alpha  = as_uint (p.w);
  uint   magnitude = alpha & 0x7fffffffU;
  float  r      = mask[index] * radius + 0.5f;
  float  r2     = r * r;
  float  weight = 1.0f;

  if (factor != 0.0f)
    {
      float v = dot (p.xyz, luminance.xyz);

      v = (v - threshold_low) / (threshold_high - threshold_low);
      weight = v <= 0.0f ? 1.0f :
               v >= 1.0f ? highlight_max : exp (v * factor);
    }

  weight /= M_PI_F * r2;
  p.w    *= weight;
  p.xyz  *= p.w;

  /* Tiny alpha can be flushed to zero by device arithmetic, and negative
   * alpha can leave a cancellation residual instead of the CPU's zero.
   * Also bound each contribution so the compensated sum stays finite.
   */
  if (fallback &&
      ((magnitude && (alpha != magnitude || magnitude < as_uint (0x1p-100f))) ||
       !all (isfinite (p)) || !isfinite (weight) || !isfinite (r2) ||
       any (fabs (p) > FLT_MAX * 0x1p-24f)))
    atomic_or (fallback, 1);

  pixels[target] = p;
  disc[target]   = (float2) (weight, r2);
}

__kernel void
focus_lens_prepare (__global const float4 *src,
                    __global const float  *mask,
                    __global       float4 *pixels,
                    __global       float2 *disc,
                    float                 radius,
                    float4                luminance,
                    float                 threshold_low,
                    float                 threshold_high,
                    float                 factor,
                    float                 highlight_max)
{
  prepare_pixel (src, mask, pixels, disc, radius, luminance, threshold_low,
                 threshold_high, factor, highlight_max, 0);
}

__kernel void
focus_lens_prepare_checked (__global const float4 *src,
                            __global const float  *mask,
                            __global       float4 *pixels,
                            __global       float2 *disc,
                            float                 radius,
                            float4                luminance,
                            float                 threshold_low,
                            float                 threshold_high,
                            float                 factor,
                            float                 highlight_max,
                            volatile __global int *fallback)
{
  prepare_pixel (src, mask, pixels, disc, radius, luminance, threshold_low,
                 threshold_high, factor, highlight_max, fallback);
}

/* The interval endpoints can have large opposing weights. Keep the
 * roundoff from each sum separately to preserve small contributions.
 */
static void
add_pixel (float4 *hi, float4 *lo, float4 value)
{
  float4 sum = *hi + value;
  float4 b   = sum - *hi;

  *lo += (*hi - (sum - b)) + (value - b);
  *hi  = sum;
}

static void
add_endpoint (__global float4 *delta,
              __global float4 *delta_w,
              int             count,
              int             index,
              float4          pixel,
              float           weight,
              int             active)
{
  float4 hi = delta[index];
  float4 lo = delta[count + index];
  float4 w  = delta_w[index];
  float  s  = w.x + weight;
  float  b  = s - w.x;

  add_pixel (&hi, &lo, pixel);

  w.y += (w.x - (s - b)) + (weight - b);
  w.x  = s;
  w.z += active;

  delta[index]         = hi;
  delta[count + index] = lo;
  delta_w[index]       = w;
}

/* The host truncates a correctly rounded sqrtf to find the span. An
 * approximate device sqrt can fall on the other side of an integer.
 * Near that boundary, compare against the midpoint before rounding.
 */
static int
disc_span (float r2)
{
  float root     = sqrt (r2);
  int   span     = convert_int_rtz (root);
  float fraction = root - span;
  int   near     = span + (fraction > 0.5f);

  if (near > 0 && fabs (root - near) < 0.001f)
    {
      float boundary = (float) near;
      float prev     = as_float (as_uint (boundary) - 1);

      /* The squared half-step is below the spacing of the radicand. */
      return near - ((float) (near * near) - r2 >=
                     boundary * (boundary - prev));
    }

  return span;
}

/* Each work item blurs a horizontal strip of one output row. A source
 * disc contributes at the two endpoints of its intersection with the row.
 * The prefix sum of those endpoints gives the weighted pixels and alpha.
 */
__kernel void
focus_lens_disc (__global const float4 *pixels,
                 __global const float2 *disc,
                 __global       float4 *delta,
                 __global       float4 *delta_w,
                 __global       float4 *dst,
                 __constant const int  *spans,
                 int4                   source,
                 int                    width,
                 int                    iradius,
                 int                    strip_width)
{
  int    y         = get_global_id (0);
  int    rows      = get_global_size (0);
  int    left      = get_global_id (1) * strip_width;
  int    right     = min (width, left + strip_width);
  int    count     = width * rows;
  int    cy        = y + source.w;
  int    first     = max (-iradius, -cy);
  int    last      = min (iradius, source.y - 1 - cy);
  float4 sum       = (float4) (0.0f);
  float4 sum_lo    = (float4) (0.0f);
  float2 weight    = (float2) (0.0f);
  int    active    = 0;

  for (int x = left; x < right; x++)
    {
      int index = x * rows + y;

      delta[index]         = (float4) (0.0f);
      delta[count + index] = (float4) (0.0f);
      delta_w[index]       = (float4) (0.0f);
    }

  for (int r = first; r <= last; r++)
    {
      int   span    = spans[abs (r)];
      int   x1      = max (left - span, -source.z);
      int   x2      = min (right + span, source.x - source.z);
      float r2      = r * r;
      int   src_row = cy + r;

      for (int x = x1; x < x2; x++)
        {
          int    index = (x + source.z) * source.y + src_row;
          float2 d     = disc[index];
          float4 p;
          int    s, xl, xr, nonzero;

          if (d.y < r2)
            continue;

          s  = disc_span (d.y - r2);
          xl = max (left, x - s);
          xr = x + s + 1;

          if (xl >= xr || xl >= right)
            continue;

          p       = pixels[index];
          nonzero = p.w != 0.0f;
          add_endpoint (delta, delta_w, count, xl * rows + y, p, d.x, nonzero);

          if (xr < right)
            add_endpoint (delta, delta_w, count, xr * rows + y,
                          -p, -d.x, -nonzero);
        }
    }

  for (int x = left; x < right; x++)
    {
      int    index = x * rows + y;
      float4 w     = delta_w[index];
      float  s     = weight.x + w.x;
      float  b     = s - weight.x;
      float4 p;

      add_pixel (&sum, &sum_lo, delta[index]);
      sum_lo += delta[count + index];
      weight.y += (weight.x - (s - b)) + (w.x - b) + w.y;
      weight.x  = s;
      active   += convert_int_rtz (w.z);

      /* Cancellation of large endpoints can leave a tiny nonzero sum
       * even where every contributing sample is fully transparent.
       */
      p     = active ? sum + sum_lo : (float4) (0.0f);
      p.xyz = p.xyz / p.w;
      p.w   = p.w / (weight.x + weight.y);

      dst[y * width + x] = p;
    }
}
