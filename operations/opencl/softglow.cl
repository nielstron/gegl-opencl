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

static float
linear_to_gamma (const float value)
{
  if (value > 0.003130804954f)
    return 1.055f * pow (value, 1.0f / 2.4f) - 0.055f;

  return 12.92f * value;
}

static float
gamma_to_linear (const float value)
{
  if (value > 0.04045f)
    return pow ((value + 0.055f) / 1.055f, 2.4f);

  return value / 12.92f;
}

/* YA float -> sigmoidal transfer of Y', as the Y float pixels that
 * gegl:gaussian-blur blurs in softglow.c
 */
__kernel void softglow_sigmoid (const global float2 *in,
                                      global float  *out,
                                const        float   slope,
                                const        float   brightness)
{
  const int gid = get_global_id (0);
  float     val = linear_to_gamma (in[gid].x);

  val = 1.0f / (1.0f + exp (-slope * (val - 0.5f)));
  val = clamp (val * brightness, 0.0f, 1.0f);

  out[gid] = gamma_to_linear (val);
}

/* Screens a chunk of RGBA float pixels with the blurred image, at
 * (x0, y0) of the blurred image, which is stride pixels wide.
 */
__kernel void softglow_process (const global float4 *in,
                                      global float4 *out,
                                const global float  *blur,
                                const        int     stride,
                                const        int     x0,
                                const        int     y0,
                                const        int     width)
{
  const int    gid  = get_global_id (0);
  const float  b    = linear_to_gamma (blur[(y0 + gid / width) * stride +
                                            x0 + gid % width]);
  const float4 v    = in[gid];
  const float3 tmp  = (1.0f - v.xyz) * (1.0f - b);

  out[gid] = (float4) (clamp (1.0f - tmp, 0.0f, 1.0f), v.w);
}
