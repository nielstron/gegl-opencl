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

/* One iteration of the mean curvature flow.  The output pixel (x, y) is
 * computed from the source pixel (x + 1, y + 1), so the valid region
 * shrinks by one pixel on each side per iteration, exactly like in the
 * CPU implementation.
 *
 * The CPU version computes, in double precision,
 *
 *   0.25 * |g| * (dx^2 dyy + dy^2 dxx - 2 dx dy dxy) / |g|^3
 *
 * Here the equivalent
 *
 *   0.25 * (nx^2 dyy + ny^2 dxx - 2 nx ny dxy) / (nx^2 + ny^2)
 *
 * is used, where (nx, ny) is the gradient divided by max (|dx|, |dy|), so
 * that nothing overflows or underflows in single precision.
 */

__kernel void
mean_curvature_blur (__global const float4 *src,
                     int                    src_stride,
                     __global       float4 *dst,
                     int                    dst_stride)
{
  int x = get_global_id (0);
  int y = get_global_id (1);

  __global const float4 *c = src + (y + 1) * src_stride + (x + 1);

  float4 center      = c[0];
  float4 left        = c[-1];
  float4 right       = c[1];
  float4 top         = c[-src_stride];
  float4 bottom      = c[src_stride];
  float4 topleft     = c[-src_stride - 1];
  float4 topright    = c[-src_stride + 1];
  float4 bottomleft  = c[src_stride - 1];
  float4 bottomright = c[src_stride + 1];

  float4 dx   = right - left;
  float4 dy   = bottom - top;
  float4 s    = fmax (fabs (dx), fabs (dy));
  int4   flat = s == 0.0f;
  float4 inv  = 1.0f / select (s, (float4) (1.0f), flat);
  float4 nx   = dx * inv;
  float4 ny   = dy * inv;
  float4 nx2  = nx * nx;
  float4 ny2  = ny * ny;

  float4 dxx = right + left - 2.0f * center;
  float4 dyy = bottom + top - 2.0f * center;
  float4 dxy = 0.25f * (bottomright - topright - bottomleft + topleft);

  float4 n = nx2 * dyy + ny2 * dxx - 2.0f * nx * ny * dxy;
  float4 d = select (nx2 + ny2, (float4) (1.0f), flat);

  float4 result = center + select (0.25f * n / d, (float4) (0.0f), flat);
  result.w = center.w;

  dst[y * dst_stride + x] = result;
}
