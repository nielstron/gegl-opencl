/* This file is an image processing operation for GEGL
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

/* Mirrors the CPU implementation step by step, as compiled for arm64
 * (where the C compiler contracts multiply-adds into fma), so that the same
 * source pixels are visited with the same weights:
 *
 *  - the per-column and per-row offsets to the end point, which the CPU
 *    computes in double precision, are computed on the host and passed in
 *    the deltas buffer (deltas[x] for columns, deltas[width + y] for rows);
 *  - the number of samples, ceil (sqrt (len^2) + 1) on the CPU, uses an
 *    exact integer square root instead of the (inexact) OpenCL sqrt ();
 *  - 1 / n is corrected to the correctly rounded value;
 *  - sample coordinates are truncated, not floored, like the CPU does.
 *
 * Automatic contraction into fma is disabled; fma () is used explicitly
 * where the CPU code uses it.
 */

#pragma OPENCL FP_CONTRACT OFF

#define MAX_NUM_IT     200
#define NOMINAL_NUM_IT 100

static float4
get_pixel_color (__global const float4 *in_buf,
                 int                    src_x,
                 int                    src_y,
                 int                    src_width,
                 int                    src_height,
                 int                    x,
                 int                    y)
{
  int ix = clamp (x - src_x, 0, src_width  - 1);
  int iy = clamp (y - src_y, 0, src_height - 1);

  return in_buf[iy * src_width + ix];
}

/* is k * k >= s, exactly? */
static int
square_ge (int   k,
           float s)
{
  if (s >= 8388608.0f) /* s is an integer */
    return (long) k * (long) k >= (long) s;
  else                 /* k * k < 2^24 when it matters, exact in float */
    return (float) min (k * k, 16777216) >= s;
}

/* smallest integer k with k * k >= s, i.e. ceil (sqrt (s)) */
static int
ceil_sqrt (float s)
{
  int k = (int) ceil (sqrt (s));

  while (k > 0 && square_ge (k - 1, s))
    k--;
  while (! square_ge (k, s))
    k++;

  return k;
}

/* largest integer r with r * r <= v */
static int
floor_sqrt (int v)
{
  int r = (int) sqrt ((float) v);

  while (r * r > v)
    r--;
  while ((r + 1) * (r + 1) <= v)
    r++;

  return r;
}

/* correctly rounded 1 / n */
static float
reciprocal (int n)
{
  float fn   = n;
  float r    = 1.0f / fn;
  float best = r;
  float err  = fabs (fma (-r, fn, 1.0f));
  float c;

  c = nextafter (r, 0.0f);
  if (fabs (fma (-c, fn, 1.0f)) < err)
    {
      best = c;
      err  = fabs (fma (-c, fn, 1.0f));
    }

  c = nextafter (r, 2.0f);
  if (fabs (fma (-c, fn, 1.0f)) < err)
    best = c;

  return best;
}

__kernel void
motion_blur_zoom (__global const float4 *in_buf,
                  int                    src_x,
                  int                    src_y,
                  int                    src_width,
                  int                    src_height,
                  __global       float4 *out_buf,
                  int                    dst_x,
                  int                    dst_y,
                  __global const float  *deltas)
{
  int gidx = get_global_id (0);
  int gidy = get_global_id (1);

  float4 sum = (float4) (0.0f);

  float ex = deltas[gidx];
  float ey = deltas[get_global_size (0) + gidy];

  int   xy_len = max (ceil_sqrt (fma (ex, ex, ey * ey)) + 1, 3);
  float inv_xy_len, dxx, dyy, ix, iy;
  int   i;

  if (xy_len > NOMINAL_NUM_IT)
    xy_len = min (NOMINAL_NUM_IT + floor_sqrt (xy_len - NOMINAL_NUM_IT),
                  MAX_NUM_IT);

  inv_xy_len = reciprocal (xy_len);

  dxx = ex * inv_xy_len;
  dyy = ey * inv_xy_len;

  ix = gidx + dst_x;
  iy = gidy + dst_y;

  for (i = 0; i < xy_len; i++)
    {
      float  dx = ix - floor (ix);
      float  dy = iy - floor (iy);
      int    x0 = (int) ix;
      int    y0 = (int) iy;
      int    x1 = (int) (ix + 1.0f);
      int    y1 = (int) (iy + 1.0f);
      float4 pix0, pix1, pix2, pix3, mixy0, mixy1;

      pix0 = get_pixel_color (in_buf, src_x, src_y, src_width, src_height, x0, y0);
      pix1 = get_pixel_color (in_buf, src_x, src_y, src_width, src_height, x1, y0);
      pix2 = get_pixel_color (in_buf, src_x, src_y, src_width, src_height, x0, y1);
      pix3 = get_pixel_color (in_buf, src_x, src_y, src_width, src_height, x1, y1);

      mixy0 = fma ((float4) (dy), pix2 - pix0, pix0);
      mixy1 = fma ((float4) (dy), pix3 - pix1, pix1);

      sum += fma ((float4) (dx), mixy1 - mixy0, mixy0);

      ix += dxx;
      iy += dyy;
    }

  out_buf[gidy * get_global_size (0) + gidx] = sum * inv_xy_len;
}
