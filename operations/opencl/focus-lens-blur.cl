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

static float
focus_lens_radius (float mask,
                   float radius)
{
#if FOCUS_LENS_USE_FMA
  return fma (mask, radius, 0.5f);
#else
  return mask * radius + 0.5f;
#endif
}

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
  float  r      = focus_lens_radius (mask[index], radius);
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
                 __global       float4 *dst,
                 __constant const int  *spans,
                 int4                   source,
                 int                    width,
                 int                    iradius,
                 int                    strip_width,
                 __global       float4 *delta_w)
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

/* One flag per lane checks the complete input halo of a tile. Striding
 * by the work size lets adjacent lanes read adjacent input pixels.
 */
__kernel void
focus_lens_check_opacity (__global const float4 *src,
                          __global       uchar  *flags,
                          int                    count)
{
  int lane = get_global_id (0);

  for (int i = lane; i < count; i += get_global_size (0))
    if (src[i].w != 1.0f)
      {
        flags[lane] = 1;
        return;
      }

  flags[lane] = 0;
}

/* For opaque sRGB u8, one uint4 holds the weighted RGB and the weight.
 * The Q28 sum is below 16. At Chebyshev distance k there are at most
 * 8k samples with weight at most about 1/(pi*k*k). Including the center,
 * the bound through distance 32 is less than 12.
 */
__kernel void
focus_lens_prepare_opaque (__global const float4 *src,
                           __global const float  *mask,
                           __global       uint4  *pixels,
                           __global       float  *radius2,
                           float                  radius)
{
  int    x      = get_global_id (0);
  int    y      = get_global_id (1);
  int    index  = y * get_global_size (0) + x;
  int    target = x * get_global_size (1) + y;
  float  r      = focus_lens_radius (mask[index], radius);
  float  r2     = r * r;
  float  weight = 1.0f / (M_PI_F * r2);
  float4 moment = (float4) (src[index].xyz * weight, weight);

  pixels[target]  = convert_uint4_rte (moment * 0x1p28f);
  radius2[target] = r2;
}

__kernel void
focus_lens_disc_opaque (__global const uint4 *pixels,
                        __global const float *radius2,
                        __global       uint4 *delta,
                        __global       float4 *dst,
                        __constant const int *spans,
                        int4                  source,
                        int                   width,
                        int                   iradius,
                        int                   strip_width)
{
  int   y     = get_global_id (0);
  int   rows  = get_global_size (0);
  int   left  = get_global_id (1) * strip_width;
  int   right = min (width, left + strip_width);
  int   cy    = y + source.w;
  int   first = max (-iradius, -cy);
  int   last  = min (iradius, source.y - 1 - cy);
  uint4 sum   = (uint4) (0);

  for (int x = left; x < right; x++)
    delta[x * rows + y] = (uint4) (0);

  for (int r = first; r <= last; r++)
    {
      int   span    = spans[abs (r)];
      int   x1      = max (left - span, -source.z);
      int   x2      = min (right + span, source.x - source.z);
      float r2      = r * r;
      float min_r2  = max (r2, 1.0f);
      int   src_row = cy + r;

      for (int x = x1; x < x2; x++)
        {
          int    index = (x + source.z) * source.y + src_row;
          float  d2    = radius2[index];
          int    s, xl, xr;
          uint4  p;

          if (d2 < min_r2)
            continue;

          s  = disc_span (d2 - r2);
          xl = max (left, x - s);
          xr = x + s + 1;

          if (xl >= xr || xl >= right)
            continue;

          p = pixels[index];

          /* Endpoint differences wrap modulo 2^32. The prefix recovers
           * the nonnegative sum within the bound above. */
          delta[xl * rows + y] += p;
          if (xr < right)
            delta[xr * rows + y] -= p;
        }
    }

  for (int x = left; x < right; x++)
    {
      int    self = (x + source.z) * source.y + cy;
      uint4  p;
      float4 v;

      sum += delta[x * rows + y];
      p    = sum;
      if (radius2[self] < 1.0f)
        p += pixels[self];

      v = convert_float4 (p);
      dst[y * width + x] = (float4) (v.xyz / v.w, 1.0f);
    }
}
