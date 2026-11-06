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

/* The input has a border of 3 pixels around the output region, as
 * requested by the operation.
 */
#define BORDER 3

/* sRGB transfer function, see babl/base/util.h */
static float
linear_to_gamma (float value)
{
  if (value > 0.003130804954f)
    return 1.055f * powr (value, 1.0f / 2.4f) - 0.055f;
  return 12.92f * value;
}

static float
shade_at (float m[3][3],
          float Lx,
          float Ly,
          float Lz,
          float Nz2,
          float NzLz)
{
  float Nx = m[0][0] + m[1][0] + m[2][0] - m[0][2] - m[1][2] - m[2][2];
  float Ny = m[2][0] + m[2][1] + m[2][2] - m[0][0] - m[0][1] - m[0][2];
  float NdotL;

  if (Nx == 0.0f && Ny == 0.0f)
    return Lz;

  NdotL = Nx * Lx + Ny * Ly + NzLz;

  if (NdotL < 0.0f)
    return 0.0f;

  return NdotL / sqrt (Nx * Nx + Ny * Ny + Nz2);
}

__kernel void
emboss_ya (__global const float2 *in,
           __global       float2 *out,
                          float   Lx,
                          float   Ly,
                          float   Lz,
                          float   Nz2,
                          float   NzLz,
                          int     to_gamma)
{
  int   x      = get_global_id (0);
  int   y      = get_global_id (1);
  int   width  = get_global_size (0);
  int   stride = width + 2 * BORDER;
  int   center = (y + BORDER) * stride + x + BORDER;
  float m[3][3];
  int   i, j;

  for (i = 0; i < 3; i++)
    for (j = 0; j < 3; j++)
      {
        float2 v = in[center + (i - 1) * stride + (j - 1)];

        m[i][j] = v.y * v.x;
      }

  float shade = shade_at (m, Lx, Ly, Lz, Nz2, NzLz);

  if (to_gamma)
    shade = linear_to_gamma (shade);

  out[y * width + x] = (float2) (shade, in[center].y);
}

__kernel void
emboss_rgba (__global const float4 *in,
             __global       float4 *out,
                            float   Lx,
                            float   Ly,
                            float   Lz,
                            float   Nz2,
                            float   NzLz,
                            int     to_gamma)
{
  int    x      = get_global_id (0);
  int    y      = get_global_id (1);
  int    width  = get_global_size (0);
  int    stride = width + 2 * BORDER;
  int    center = (y + BORDER) * stride + x + BORDER;
  float  m[3][3];
  float4 c;
  int    i, j;

  for (i = 0; i < 3; i++)
    for (j = 0; j < 3; j++)
      {
        float4 v = in[center + (i - 1) * stride + (j - 1)];

        /* summed per channel like the CPU code, to round alike */
        m[i][j] = v.w * v.x;
        m[i][j] += v.w * v.y;
        m[i][j] += v.w * v.z;
      }

  c      = in[center];
  c.xyz *= shade_at (m, Lx, Ly, Lz, Nz2, NzLz);

  if (to_gamma)
    c.xyz = (float3) (linear_to_gamma (c.x),
                      linear_to_gamma (c.y),
                      linear_to_gamma (c.z));

  out[y * width + x] = c;
}
