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

/* Quantized (256 bins) median / percentile blur of 4-component
 * (color + alpha) pixels, with the same selection semantics as the CPU
 * histogram implementation:
 *
 *  - every component is quantized to bin = floor (clamp (v) * 255 + 0.5);
 *  - color bins are weighted by the alpha bin of their pixel, alpha bins
 *    have weight 1;
 *  - the result is the smallest bin whose cumulative weight reaches
 *    max (ceil (total_weight * percentile), 1), or 0 for a zero total.
 *
 * Instead of sliding a histogram, each work item finds that bin with a
 * binary search over the 256 bins (8 counting passes over the
 * neighborhood), for all four components at once.  The neighborhood bins
 * of the work group are cached in local memory.
 *
 * ceil (count * percentile) is evaluated exactly the way the CPU does it,
 * in double precision, using 64-bit integer arithmetic: the percentile is
 * passed as mantissa / 2^shift.
 */

#pragma OPENCL FP_CONTRACT OFF

/* ceil (fl64 (count * (mantissa / 2^shift))), count > 0, mantissa > 0 */
static int
ceil_scaled (uint  count,
             ulong mantissa,
             int   shift)
{
  ulong lo = (ulong) count * mantissa;
  ulong hi = mul_hi ((ulong) count, mantissa);
  int   bits;
  int   d;
  ulong m;

  bits = hi ? 128 - (int) clz (hi) : 64 - (int) clz (lo);
  d    = max (bits - 53, 0);

  /* round the product to 53 significant bits, ties to even */
  if (d == 0)
    {
      m = lo;
    }
  else
    {
      ulong rem  = lo & ((1UL << d) - 1);
      ulong halfway = 1UL << (d - 1);

      m = (lo >> d) | (hi << (64 - d));

      if (rem > halfway || (rem == halfway && (m & 1)))
        m++;
    }

  /* value = m * 2^d / 2^shift */
  shift -= d;

  if (shift <= 0)
    return (int) (m << -shift);
  if (shift >= 64)
    return 1;

  return (int) ((m >> shift) + ((m & ((1UL << shift) - 1)) != 0));
}

static int
target_count (uint  count,
              ulong mantissa,
              int   shift)
{
  if (count == 0)
    return 0;
  if (mantissa == 0)
    return 1;

  return max (ceil_scaled (count, mantissa, shift), 1);
}

/* correctly rounded bin / 255.0f */
static float
bin_value (int bin)
{
  float n    = bin;
  float q    = n / 255.0f;
  float best = q;
  float err  = fabs (fma (-q, 255.0f, n));
  float c;

  c = nextafter (q, 0.0f);
  if (fabs (fma (-c, 255.0f, n)) < err)
    {
      best = c;
      err  = fabs (fma (-c, 255.0f, n));
    }

  c = nextafter (q, 2.0f);
  if (fabs (fma (-c, 255.0f, n)) < err)
    best = c;

  return best;
}

static uchar4
quantize (float4 v)
{
  /* SAFE_CLAMP () of the CPU code, which also maps NaN to 0 */
  v = select ((float4) (0.0f), fmin (v, 1.0f), v > 0.0f);

  return convert_uchar4 (floor (fma (v, (float4) (255.0f), (float4) (0.5f))));
}

__kernel void
median_blur (__global const float4 *src,
             __global       float4 *dst,
             __local        uchar4 *cache,
             __constant     int    *outline,
             int                    width,
             int                    height,
             int                    radius,
             ulong                  color_mantissa,
             int                    color_shift,
             int                    alpha_target)
{
  int lx         = get_local_id (0);
  int ly         = get_local_id (1);
  int lw         = get_local_size (0);
  int lh         = get_local_size (1);
  int x0         = get_group_id (0) * lw;
  int y0         = get_group_id (1) * lh;
  int x          = x0 + lx;
  int y          = y0 + ly;
  int src_width  = width  + 2 * radius;
  int src_height = height + 2 * radius;
  int cw         = lw + 2 * radius;
  int ch         = lh + 2 * radius;
  int i, j;

  /* load and quantize the source pixels of the work group */
  for (j = ly; j < ch; j += lh)
    for (i = lx; i < cw; i += lw)
      {
        int sx = min (x0 + i, src_width  - 1);
        int sy = min (y0 + j, src_height - 1);

        cache[j * cw + i] = quantize (src[sy * src_width + sx]);
      }

  barrier (CLK_LOCAL_MEM_FENCE);

  if (x >= width || y >= height)
    return;

  {
    __local const uchar4 *center = cache + (ly + radius) * cw + (lx + radius);
    uint  count = 0;
    int4  target;
    int4  lo    = (int4) (0);
    int4  hi    = (int4) (255);
    int   pass;
    int   dy, dx;
    float4 result;

    for (dy = -radius; dy <= radius; dy++)
      {
        int                   w   = outline[abs (dy)];
        __local const uchar4 *row = center + dy * cw;

        for (dx = -w; dx <= w; dx++)
          count += row[dx].w;
      }

    target.xyz = (int3) (target_count (count, color_mantissa, color_shift));
    target.w   = alpha_target;

    for (pass = 0; pass < 8; pass++)
      {
        int4 mid = (lo + hi) >> 1;
        int4 sum = (int4) (0);

        for (dy = -radius; dy <= radius; dy++)
          {
            int                   w   = outline[abs (dy)];
            __local const uchar4 *row = center + dy * cw;

            for (dx = -w; dx <= w; dx++)
              {
                int4 b      = convert_int4 (row[dx]);
                int4 weight = (int4) (b.w, b.w, b.w, 1);

                /* (b <= mid) is -1 (all bits set) for true */
                sum += (b <= mid) & weight;
              }
          }

        {
          int4 found = sum >= target;

          hi = select (hi, mid, found);
          lo = select (mid + 1, lo, found);
        }
      }

    result.x = bin_value (lo.x);
    result.y = bin_value (lo.y);
    result.z = bin_value (lo.z);
    result.w = bin_value (lo.w);

    if (count == 0)
      result.xyz = (float3) (0.0f);

    dst[y * width + x] = result;
  }
}
