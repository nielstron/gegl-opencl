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

#define ALPHA_FLOOR (1.0f / 65536.0f)

/* babl_epsilon_for_zero_float () */
static float
epsilon_for_zero (const float value)
{
  return (value > ALPHA_FLOOR || value < -ALPHA_FLOOR) ? value : ALPHA_FLOOR;
}

static float
linear_to_gamma (const float value)
{
  if (value > 0.003130804954f)
    return 1.055f * pow (value, 1.0f / 2.4f) - 0.055f;

  return 12.92f * value;
}

/* The part of the graph of bloom.c before the blur, for RGBA float input
 * pixels: the Y float input cast from Y' float, gegl:levels,
 * gegl:rgb-clip and gegl:multiply with the input, as the RaGaBaA float
 * pixels that gegl:gaussian-blur blurs.
 */
__kernel void bloom_glow (const global float4 *in,
                                global float4 *out,
                          const        float4  luminance,
                          const        float   in_offset,
                          const        float   scale,
                          const        float   out_offset,
                          const        int     clip_low,
                          const        float   low_limit,
                          const        int     clip_high,
                          const        float   high_limit)
{
  const int    gid = get_global_id (0);
  const float4 v   = in[gid];
  float        g;

  g = linear_to_gamma (v.x * luminance.x +
                       v.y * luminance.y +
                       v.z * luminance.z);

  g = (g - in_offset) * scale + out_offset;

  if (clip_high && g > high_limit)
    g = high_limit;
  else if (clip_low && g < low_limit)
    g = low_limit;

  out[gid] = (float4) (v.xyz * g * epsilon_for_zero (v.w), v.w);
}

/* gegl:add (screen = 0) or gegl:screen (screen = 1) of the input and the
 * blurred glow at (x0, y0) + the chunk position, in images that are
 * stride pixels wide; writes RGBA float or RaGaBaA float.
 */
__kernel void bloom_combine (const global float4 *in,
                             const global float4 *blur,
                                   global float4 *out,
                             const        int     stride,
                             const        int     x0,
                             const        int     y0,
                             const        int     width,
                             const        int     screen)
{
  const int    gid = get_global_id (0);
  const int    idx = (y0 + gid / width) * stride + x0 + gid % width;
  const float4 v   = in[idx];
  const float4 b   = blur[idx];

  if (! screen)
    {
      out[gid] = (float4) (v.xyz + b.xyz * (1.0f / epsilon_for_zero (b.w)),
                           v.w);
    }
  else
    {
      const float  a_b = v.w;
      const float  a_a = b.w;
      const float  a_d = a_a + a_b - a_a * a_b;
      const float3 c_b = v.xyz * epsilon_for_zero (v.w);
      const float3 c   = b.xyz + c_b - b.xyz * c_b;

      /* CLAMP (c, 0, a_d) */
      out[gid] = (float4) (select (select (c, (float3) 0.0f, c < 0.0f),
                                   (float3) a_d, c > a_d),
                           a_d);
    }
}
