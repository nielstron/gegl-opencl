/* This file is part of GEGL
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
 *
 * Copyright 2012 Victor Oliveira (victormatheus@gmail.com)
 *           2013 Daniel Sabo
 */

/* Conversions from 8-bit gamma-encoded sRGB.
 *
 * gegl-cl-color.c prepends a header, generated from babl at runtime, that
 * defines BABL_LUMINANCE_RED / _GREEN / _BLUE, babl's u8 -> float table
 * babl_u8_to_float[256] and, for each kernel below that converts to RGB, a
 * table <kernel name>_lut[256] holding babl's result of converting each
 * 8-bit value to linear light.  babl picks between implementations of the
 * same conversion (single or double precision) at runtime, so the tables
 * are taken from the very conversion the CPU path would use.
 *
 * The table used for YA can not be recovered from babl's result, so there
 * are variants for the single (gamma_u8_lut_f) and double precision
 * (gamma_u8_lut_d) tables and the luminance code of babl/base and of
 * extensions/grey.c, and gegl-cl-color.c picks the one that matches babl.
 */

#pragma OPENCL FP_CONTRACT OFF

/* babl/babl.h: BABL_ALPHA_FLOOR_F */
#define BABL_ALPHA_FLOOR_F (1.0f / 65536.0f)

float babl_epsilon_for_zero (float value);
float luminance (float r, float g, float b);
float luminance_grey (float r, float g, float b);

/* babl/base/util.h */
float babl_epsilon_for_zero (float value)
{
  return (value > BABL_ALPHA_FLOOR_F || value < -BABL_ALPHA_FLOOR_F) ?
         value : BABL_ALPHA_FLOOR_F;
}

/* babl/base/model-gray.c: rgba_to_graya_float () */
float luminance (float r, float g, float b)
{
  return fma (b, BABL_LUMINANCE_BLUE,
              fma (r, BABL_LUMINANCE_RED, g * BABL_LUMINANCE_GREEN));
}

/* extensions/grey.c: conv_rgbaF_linear_yaF_linear () */
float luminance_grey (float r, float g, float b)
{
  return fma (b, BABL_LUMINANCE_BLUE,
              fma (g, BABL_LUMINANCE_GREEN, r * BABL_LUMINANCE_RED));
}

/* R'G'B' u8 -> RGBA float */
__kernel void rgb_gamma_u8_to_rgbaf (__global const uchar  * in,
                                     __global       float4 * out)
{
  int gid = get_global_id(0);

  float r = rgb_gamma_u8_to_rgbaf_lut[in[3 * gid + 0]];
  float g = rgb_gamma_u8_to_rgbaf_lut[in[3 * gid + 1]];
  float b = rgb_gamma_u8_to_rgbaf_lut[in[3 * gid + 2]];

  out[gid] = (float4) (r, g, b, 1.0f);
}

/* R'G'B' u8 -> RaGaBaA float */
__kernel void rgb_gamma_u8_to_ragabaf (__global const uchar  * in,
                                       __global       float4 * out)
{
  int gid = get_global_id(0);

  float r = rgb_gamma_u8_to_ragabaf_lut[in[3 * gid + 0]];
  float g = rgb_gamma_u8_to_ragabaf_lut[in[3 * gid + 1]];
  float b = rgb_gamma_u8_to_ragabaf_lut[in[3 * gid + 2]];

  out[gid] = (float4) (r, g, b, 1.0f);
}

/* R'G'B' u8 -> YA float */
#define RGB_GAMMA_U8_TO_YAF(name, lut, luminance)                   \
__kernel void name (__global const uchar  * in,                     \
                    __global       float2 * out)                    \
{                                                                   \
  int gid = get_global_id(0);                                       \
                                                                    \
  float r = lut[in[3 * gid + 0]];                                   \
  float g = lut[in[3 * gid + 1]];                                   \
  float b = lut[in[3 * gid + 2]];                                   \
                                                                    \
  out[gid] = (float2) (luminance (r, g, b), 1.0f);                  \
}

RGB_GAMMA_U8_TO_YAF (rgb_gamma_u8_to_yaf, gamma_u8_lut_f, luminance)
RGB_GAMMA_U8_TO_YAF (rgb_gamma_u8_to_yaf_grey, gamma_u8_lut_f, luminance_grey)
RGB_GAMMA_U8_TO_YAF (rgb_gamma_u8_to_yaf_d, gamma_u8_lut_d, luminance)
RGB_GAMMA_U8_TO_YAF (rgb_gamma_u8_to_yaf_d_grey, gamma_u8_lut_d, luminance_grey)

/* R'G'B'A u8 -> RGBA float */
__kernel void rgba_gamma_u8_to_rgbaf (__global const uchar4 * in,
                                      __global       float4 * out)
{
  int gid = get_global_id(0);
  uchar4 in_v = in[gid];

  float r = rgba_gamma_u8_to_rgbaf_lut[in_v.x];
  float g = rgba_gamma_u8_to_rgbaf_lut[in_v.y];
  float b = rgba_gamma_u8_to_rgbaf_lut[in_v.z];

  out[gid] = (float4) (r, g, b, babl_u8_to_float[in_v.w]);
}

/* R'G'B'A u8 -> RaGaBaA float */
__kernel void rgba_gamma_u8_to_ragabaf (__global const uchar4 * in,
                                        __global       float4 * out)
{
  int gid = get_global_id(0);
  uchar4 in_v  = in[gid];
  float4 out_v;
  float4 tmp_v;

  tmp_v = (float4)(rgba_gamma_u8_to_ragabaf_lut[in_v.x],
                   rgba_gamma_u8_to_ragabaf_lut[in_v.y],
                   rgba_gamma_u8_to_ragabaf_lut[in_v.z],
                   babl_u8_to_float[in_v.w]);

  out_v   = tmp_v * babl_epsilon_for_zero (tmp_v.w);
  out_v.w = tmp_v.w;

  out[gid] = out_v;
}

/* R'G'B'A u8 -> YA float */
#define RGBA_GAMMA_U8_TO_YAF(name, lut, luminance)                  \
__kernel void name (__global const uchar4 * in,                     \
                    __global       float2 * out)                    \
{                                                                   \
  int gid = get_global_id(0);                                       \
  uchar4 in_v  = in[gid];                                           \
                                                                    \
  float r = lut[in_v.x];                                            \
  float g = lut[in_v.y];                                            \
  float b = lut[in_v.z];                                            \
                                                                    \
  out[gid] = (float2) (luminance (r, g, b), babl_u8_to_float[in_v.w]); \
}

RGBA_GAMMA_U8_TO_YAF (rgba_gamma_u8_to_yaf, gamma_u8_lut_f, luminance)
RGBA_GAMMA_U8_TO_YAF (rgba_gamma_u8_to_yaf_grey, gamma_u8_lut_f, luminance_grey)
RGBA_GAMMA_U8_TO_YAF (rgba_gamma_u8_to_yaf_d, gamma_u8_lut_d, luminance)
RGBA_GAMMA_U8_TO_YAF (rgba_gamma_u8_to_yaf_d_grey, gamma_u8_lut_d, luminance_grey)
