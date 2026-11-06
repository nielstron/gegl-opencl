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

/* sRGB transfer function, see babl/base/util.h */
static float
linear_to_gamma (float value)
{
  if (value > 0.003130804954f)
    return 1.055f * powr (value, 1.0f / 2.4f) - 0.055f;
  return 12.92f * value;
}

static float
height_at (__global const float2 *in,
                          int     index,
                          int     to_gamma)
{
  float h = in[index].x;

  return to_gamma ? linear_to_gamma (h) : h;
}

/* in is YA (linear, converted here when to_gamma is set) or Y'A, with a
 * one pixel border around the output region.
 */
__kernel void
normal_map (__global const float2 *in,
            __global       float4 *out,
                           float   scale,
                           float   x_scale,
                           float   y_scale,
                           float   z_scale,
                           float   z_base,
                           int     x_component,
                           int     y_component,
                           int     z_component,
                           int     to_gamma)
{
  int    x      = get_global_id (0);
  int    y      = get_global_id (1);
  int    width  = get_global_size (0);
  int    stride = width + 2;
  int    center = (y + 1) * stride + x + 1;
  float  l      = height_at (in, center - 1,      to_gamma);
  float  r      = height_at (in, center + 1,      to_gamma);
  float  t      = height_at (in, center - stride, to_gamma);
  float  b      = height_at (in, center + stride, to_gamma);
  float  a      = in[center].y;
  float  nx     = scale * (l - r);
  float  ny     = scale * (t - b);
  float  nz     = 1.0f / sqrt (nx * nx + ny * ny + 1.0f);
  float  c[3];

  nx *= nz;
  ny *= nz;

  c[x_component] = 0.5f   + x_scale * nx;
  c[y_component] = 0.5f   + y_scale * ny;
  c[z_component] = z_base + z_scale * nz;

  out[y * width + x] = (float4) (c[0], c[1], c[2], a);
}
