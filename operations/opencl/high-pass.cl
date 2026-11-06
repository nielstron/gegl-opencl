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

static float3
linear_to_perceptual (const float3 v)
{
  return select (12.92f * v,
                 1.055f * pow (v, 1.0f / 2.4f) - 0.055f,
                 v > 0.003130804954f);
}

static float3
perceptual_to_linear (const float3 v)
{
  return select (v / 12.92f,
                 pow ((v + 0.055f) / 1.055f, 2.4f),
                 v > 0.04045f);
}

/* gegl:gamma */
static float3
gamma (const float3 v,
       const float  value)
{
  return select (-pow (-v, value), pow (v, value), v >= 0.0f);
}

/* The graph of high-pass.c for one pixel: in and blur are the RaGaBaA
 * float input and blurred input at (x0, y0) + the chunk position, in
 * images that are stride pixels wide; out is a chunk of RGBA float.
 */
__kernel void high_pass (const global float4 *in,
                         const global float4 *blur,
                               global float4 *out,
                         const        int     stride,
                         const        int     x0,
                         const        int     y0,
                         const        int     width,
                         const        float   contrast)
{
  const int    gid = get_global_id (0);
  const int    idx = (y0 + gid / width) * stride + x0 + gid % width;
  const float4 i   = in[idx];
  const float4 b   = blur[idx];
  float3       in_pm, aux_pm, over, v;
  float        in_a, aux_a, over_a, used_a;

  /* the input as R~aG~aB~aA float */
  in_a   = i.w;
  used_a = epsilon_for_zero (in_a);
  in_pm  = linear_to_perceptual (i.xyz * (1.0f / used_a)) * used_a;

  /* gegl:invert-gamma and gegl:opacity value=0.5 of the blurred input in
   * R~G~B~A float, as R~aG~aB~aA float
   */
  v      = 1.0f - linear_to_perceptual (b.xyz *
                                        (1.0f / epsilon_for_zero (b.w)));
  aux_a  = b.w * 0.5f;
  aux_pm = v * epsilon_for_zero (aux_a);

  /* gegl:over srgb=true */
  over   = aux_pm + in_pm * (1.0f - aux_a);
  over_a = aux_a + in_a - aux_a * in_a;

  /* to RGBA float, then the gegl:gamma, gegl:add and gegl:multiply chain */
  v = perceptual_to_linear (over * (1.0f / epsilon_for_zero (over_a)));
  v = gamma (v, 0.4545f);
  v = v + -0.5f;
  v = v * contrast;
  v = v + 0.5f;
  v = gamma (v, 2.2f);

  out[gid] = (float4) (v, over_a);
}
