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
 */

/* These kernels mirror babl's float code paths, so that converting on the
 * GPU gives bit-identical results to converting with babl on the CPU.
 * Operations that threshold or index tables after a conversion are
 * sensitive to differences in the last bit.
 *
 * gegl-cl-color.c prepends a header, generated from babl at runtime, that
 * defines:
 *
 *   BABL_LUMINANCE_RED / _GREEN / _BLUE  the luminance weights of sRGB
 *   babl_u8_to_float[256]                babl's u8 -> float table
 *   rgbau8_to_rgba_gamma_f_lut[256]      babl's RGBA u8 -> R'G'B'A float
 *   BABL_CORRECTLY_ROUNDED_SQRT          when the program is built with
 *                                        -cl-fp32-correctly-rounded-divide-sqrt
 *
 * Compilers contract a * b + c within an expression into a fused
 * multiply-add by default where the target has one (clang on arm64, for
 * example), so that is what babl computes there.  The explicit fma () calls
 * below mirror those contractions and contraction is disabled for
 * everything else.  Against a babl built without fused multiply-add the
 * gamma functions differ by up to 15 ulp in [-0.1, 1.3].
 *
 * Correctly rounded division and square root are not guaranteed without
 * -cl-fp32-correctly-rounded-divide-sqrt, so they are emulated with fma ().
 *
 * babl chooses the fastest of several conversion paths at runtime, and the
 * paths can differ in the last bit.  Where that happens there is a kernel
 * variant for each path, named <kernel>_<variant>, and gegl-cl-color.c picks
 * the one that matches babl.
 */

#pragma OPENCL FP_CONTRACT OFF

/* babl/babl.h: BABL_ALPHA_FLOOR_F */
#define BABL_ALPHA_FLOOR_F (1.0f / 65536.0f)

float babl_epsilon_for_zero (float value);
float babl_div_rn_recip (float a, float b, float y);
float babl_div_rn (float a, float b);
float babl_sqrt_rn (float x);
float babl_init_newton (float x, float c0, float c1, float c2);
float babl_pow_24 (float x);
float babl_pow_1_24 (float x);
float linear_to_gamma_2_2 (float value);
float gamma_2_2_to_linear (float value);
float4 unpremultiply (float4 v);
float4 premultiply_roundtrip (float4 v);
float luminance (float r, float g, float b);
float luminance_grey (float r, float g, float b);
float linear_to_gamma_2_2_pow (float value);
float gamma_2_2_to_linear_pow (float value);
float3 rgb_to_ycbcr (float r, float g, float b);
float3 ycbcr_to_rgb (float y, float cb, float cr);
uchar4 float4_to_u8 (float4 v);
uchar3 float3_to_u8 (float3 v);
uchar  float_to_u8 (float v);
float4 u8_to_float4 (uchar4 v);

/* babl/base/util.h */
float babl_epsilon_for_zero (float value)
{
  return (value > BABL_ALPHA_FLOOR_F || value < -BABL_ALPHA_FLOOR_F) ?
         value : BABL_ALPHA_FLOOR_F;
}

/* a / b, correctly rounded, given y, the correctly rounded 1 / b
 * (Markstein), for a that is not tiny.
 */
float babl_div_rn_recip (float a, float b, float y)
{
  float q = a * y;
  float r = fma (-b, q, a);

  return fma (r, y, q);
}

/* a / b, correctly rounded, for b around 1 */
float babl_div_rn (float a, float b)
{
  /* scale tiny a, so that the residual does not flush to zero */
  float scale = fabs (a) < 0x1p-90f ? 0x1p64f : 1.0f;
  float y     = 1.0f / b;

  y = fma (fma (-b, y, 1.0f), y, y);
  return babl_div_rn_recip (a * scale, b, y) * (1.0f / scale);
}

/* sqrt (x), correctly rounded, for positive normal x.  sqrt () is within
 * 3 ulp, a Newton step brings that within an ulp and the Tuckerman test
 * picks the correctly rounded neighbour.
 */
float babl_sqrt_rn (float x)
{
#ifdef BABL_CORRECTLY_ROUNDED_SQRT
  return sqrt (x);
#else
  float s = sqrt (x);
  float sp, sm;

  s  = fma (fma (-s, s, x), 0.5f * native_recip (s), s);
  sp = as_float (as_int (s) + 1);
  sm = as_float (as_int (s) - 1);

  if (fma (s, sp, -x) < 0.0f)
    s = sp;
  else if (fma (sm, s, -x) >= 0.0f)
    s = sm;
  return s;
#endif
}

/* babl/base/pow-24.h: init_newtonf (), for positive normal x.  c1 and c2
 * are passed in already multiplied by fM_LN2 * exponent and
 * fM_LN2 * fM_LN2 * exponent * exponent, rounded in single precision as
 * babl does, and the Newton constants below (1.f+1.f/5, 1.f/5, 7.f/6.f,
 * 1.f/6.f, 0.055f - 3.0f / (1 << 24)) are spelled out in hex as well.
 */
float babl_init_newton (float x, float c0, float c1, float c2)
{
  int   bits = as_int (x);
  int   iexp = (bits >> 23) - 126;
  float y    = as_float ((bits & 0x007fffff) | 0x3f000000);

  y = 2.0f * y + (float) (iexp - 2);
  return fma (c2 * y, y, fma (c1, y, c0));
}

/* babl/base/pow-24.h: babl_pow_24f () */
float babl_pow_24 (float x)
{
  float y, yy;

  if (x > 16.0f)
    return exp (log (x) * 2.4f);

  y = babl_init_newton (x, 0.9953189663f, -0x1.106562p-3f, 0x1.a8a1a6p-7f);
#define STEP_24 \
  yy = y * y; \
  y  = fma (0x1.333334p+0f, y, -((0x1.99999ap-3f * x * yy) * (yy * yy)));
  STEP_24 STEP_24 STEP_24
#undef STEP_24
  x *= y;
  return x * x * x;
}

/* babl/base/pow-24.h: babl_pow_1_24f () */
float babl_pow_1_24 (float x)
{
  float y, yy, z;

  if (x > 1024.0f)
    return exp (log (x) * (1.0f / 2.4f));

  y = babl_init_newton (x, 0.9976800269f, -0x1.d3c0b8p-5f, 0x1.0264acp-9f);
  x = babl_sqrt_rn (x);
  z = 0x1.555556p-3f * x;
#define STEP_1_24 \
  yy = y * y; \
  y  = fma (0x1.2aaaaap+0f, y, -(z * ((yy * yy) * (yy * y))));
  STEP_1_24 STEP_1_24 STEP_1_24
#undef STEP_1_24
  return x * y;
}

/* babl/base/util.h: babl_linear_to_gamma_2_2f () */
float linear_to_gamma_2_2 (float value)
{
  if (value > 0.003130804954f)
    return fma (1.055f, babl_pow_1_24 (value), -0x1.c28efcp-5f);
  return 12.92f * value;
}

/* babl/base/util.h: babl_gamma_2_2_to_linearf () */
float gamma_2_2_to_linear (float value)
{
  /* 0x1.e54edep-1f and 0x1.3d0722p-4f are 1 / 1.055f and 1 / 12.92f */
  if (value > 0.04045f)
    return babl_pow_24 (babl_div_rn_recip (value + 0.055f, 1.055f,
                                           0x1.e54edep-1f));
  if (fabs (value) < 0x1p-90f)
    return babl_div_rn_recip (value * 0x1p64f, 12.92f, 0x1.3d0722p-4f) * 0x1p-64f;
  return babl_div_rn_recip (value, 12.92f, 0x1.3d0722p-4f);
}

/* extensions/gegl-fixups.c: conv_rgbAF_rgbaF () */
float4 unpremultiply (float4 v)
{
  float recip = babl_div_rn (1.0f, babl_epsilon_for_zero (v.w));

  return (float4) (v.xyz * recip, v.w);
}

/* (v * alpha) / alpha, the way babl computes it when its conversion path
 * goes through associated alpha: extensions/gegl-fixups.c conv_rgbaF_rgbAF ()
 * followed by conv_rgbAF_rgbaF ().
 */
float4 premultiply_roundtrip (float4 v)
{
  float used_alpha = babl_epsilon_for_zero (v.w);
  float recip      = babl_div_rn (1.0f, used_alpha);

  return (float4) ((v.xyz * used_alpha) * recip, v.w);
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

/* babl/base/model-ycbcr.c converts in double precision, so for Y'CbCr get
 * as close as possible instead of mirroring babl's single precision code.
 */
float linear_to_gamma_2_2_pow (float value)
{
  if (value > 0.003130804954f)
    return fma (1.055f, pow (value, 1.0f / 2.4f), -0.055f);
  return 12.92f * value;
}

float gamma_2_2_to_linear_pow (float value)
{
  if (value > 0.04045f)
    return pow (babl_div_rn (value + 0.055f, 1.055f), 2.4f);
  return babl_div_rn (value, 12.92f);
}

float3 rgb_to_ycbcr (float r, float g, float b)
{
  return (float3) (fma ( 0.299f,    r, fma ( 0.587f,    g,  0.114f    * b)),
                   fma (-0.168736f, r, fma (-0.331264f, g,  0.5f      * b)),
                   fma ( 0.5f,      r, fma (-0.418688f, g, -0.081312f * b)));
}

float3 ycbcr_to_rgb (float y, float cb, float cr)
{
  return (float3) (fma (1.402f, cr, y),
                   fma (-0.344136f, cb, fma (-0.71414136f, cr, y)),
                   fma (1.772f, cb, y));
}

/* extensions/simple.c: float_to_u8_x1 () */
uchar4 float4_to_u8 (float4 v)
{
  return convert_uchar4_sat (fma (v, 255.0f, 0.5f));
}

uchar3 float3_to_u8 (float3 v)
{
  return convert_uchar3_sat (fma (v, 255.0f, 0.5f));
}

uchar float_to_u8 (float v)
{
  return convert_uchar_sat (fma (v, 255.0f, 0.5f));
}

float4 u8_to_float4 (uchar4 v)
{
  return (float4) (babl_u8_to_float[v.x],
                   babl_u8_to_float[v.y],
                   babl_u8_to_float[v.z],
                   babl_u8_to_float[v.w]);
}

/* -- RGBA float/u8 -- */

/* RGBA u8 -> RGBA float */
__kernel void rgbau8_to_rgbaf (__global const uchar4 * in,
                               __global       float4 * out)
{
  int gid = get_global_id(0);
  float4 in_v  = u8_to_float4 (in[gid]);
  float4 out_v = in_v;
  out[gid] = out_v;
}

/* RGBA float -> RGBA u8 */
__kernel void rgbaf_to_rgbau8 (__global const float4 * in,
                               __global       uchar4 * out)
{
  int gid = get_global_id(0);
  float4 in_v  = in[gid];
  float4 out_v = in_v;
  out[gid] = float4_to_u8 (out_v);
}


/* RGBA float -> RGB float */
__kernel void rgbaf_to_rgbf (__global const float4 * in,
                             __global       float  * out)
{
  int gid = get_global_id(0);
  float4 in_v  = in[gid];

#if (__OPENCL_VERSION__ != CL_VERSION_1_0)
  vstore3 (in_v.xyz, gid, out);
#else
  out[3 * gid]     = in_v.x;
  out[3 * gid + 1] = in_v.y;
  out[3 * gid + 2] = in_v.z;
#endif
}

/* -- RaGaBaA float -- */

/* RGBA float -> RaGaBaA float */
__kernel void rgbaf_to_ragabaf (__global const float4 * in,
                                __global       float4 * out)
{
  int gid = get_global_id(0);
  float4 in_v = in[gid];
  float4 out_v;
  out_v   = in_v * babl_epsilon_for_zero (in_v.w);
  out_v.w = in_v.w;
  out[gid] = out_v;
}


/* RaGaBaA float -> RGBA float */
__kernel void ragabaf_to_rgbaf (__global const float4 * in,
                                __global       float4 * out)
{
  int gid = get_global_id(0);
  float4 in_v  = in[gid];
  float4 out_v;
  out_v = unpremultiply (in_v);
  out[gid] = out_v;
}

/* RGBA u8 -> RaGaBaA float */
__kernel void rgbau8_to_ragabaf (__global const uchar4 * in,
                                 __global       float4 * out)
{
  int gid = get_global_id(0);
  float4 in_v  = u8_to_float4 (in[gid]);
  float4 out_v;
  out_v   = in_v * babl_epsilon_for_zero (in_v.w);
  out_v.w = in_v.w;
  out[gid] = out_v;
}


/* RaGaBaA float -> RGBA u8 */
__kernel void ragabaf_to_rgbau8 (__global const float4 * in,
                                 __global       uchar4 * out)
{
  int gid = get_global_id(0);
  float4 in_v  = in[gid];
  float4 out_v;
  out_v = unpremultiply (in_v);
  out[gid] = float4_to_u8 (out_v);
}

/* RaGaBaA float -> RGBA_GAMMA_U8 */
__kernel void ragabaf_to_rgba_gamma_u8 (__global const float4 * in,
                                        __global       uchar4 * out)
{
  int gid = get_global_id(0);
  float4 in_v  = in[gid];
  float4 tmp_v;
  tmp_v = unpremultiply (in_v);
  float4 out_v;
  out_v = (float4)(linear_to_gamma_2_2(tmp_v.x),
                   linear_to_gamma_2_2(tmp_v.y),
                   linear_to_gamma_2_2(tmp_v.z),
                   tmp_v.w);
  out[gid] = float4_to_u8 (out_v);
}

/* RaGaBaA float -> RGB_GAMMA_U8 */
__kernel void ragabaf_to_rgb_gamma_u8 (__global const float4 * in,
                                       __global       uchar  * out)
{
  int gid = get_global_id(0);
  float4 in_v  = in[gid];
  float4 tmp_v;
  tmp_v = unpremultiply (in_v);
  float4 out_v;
  out_v = (float4)(linear_to_gamma_2_2(tmp_v.x),
                   linear_to_gamma_2_2(tmp_v.y),
                   linear_to_gamma_2_2(tmp_v.z),
                   tmp_v.w);
#if (__OPENCL_VERSION__ != CL_VERSION_1_0)
  vstore3 (float3_to_u8 (out_v.xyz), gid, out);
#else
  uchar4 sat = float4_to_u8 (out_v);
  out[3 * gid]     = sat.x;
  out[3 * gid + 1] = sat.y;
  out[3 * gid + 2] = sat.z;
#endif
}

/* -- R'G'B' float -- */

/* R'G'B' float -> RGBA float */
__kernel void rgb_gamma_f_to_rgbaf (__global const float * in,
                                    __global       float4 * out)
{
  int gid = get_global_id(0);
  float4 out_v;

#if (__OPENCL_VERSION__ != CL_VERSION_1_0)
  float3 in_v;
  in_v = vload3 (gid, in);
  out_v = (float4)(gamma_2_2_to_linear(in_v.x),
                   gamma_2_2_to_linear(in_v.y),
                   gamma_2_2_to_linear(in_v.z),
                   1.0f);
#else
  float r = in[3 * gid + 0];
  float g = in[3 * gid + 1];
  float b = in[3 * gid + 2];
  out_v = (float4)(gamma_2_2_to_linear(r),
                   gamma_2_2_to_linear(g),
                   gamma_2_2_to_linear(b),
                   1.0f);
#endif
  out[gid] = out_v;
}

/* -- R'G'B'A float -- */

/* rgba float -> r'g'b'a float */
__kernel void rgbaf_to_rgba_gamma_f (__global const float4 * in,
                                     __global       float4 * out)
{
  int gid = get_global_id(0);
  float4 in_v  = in[gid];
  float4 out_v;
  out_v = (float4)(linear_to_gamma_2_2(in_v.x),
                   linear_to_gamma_2_2(in_v.y),
                   linear_to_gamma_2_2(in_v.z),
                   in_v.w);
  out[gid] = out_v;
}

/* r'g'b'a float -> rgba float */
__kernel void rgba_gamma_f_to_rgbaf (__global const float4 * in,
                                     __global       float4 * out)
{
  int gid = get_global_id(0);
  float4 in_v  = in[gid];
  float4 out_v;
  out_v = (float4)(gamma_2_2_to_linear(in_v.x),
                   gamma_2_2_to_linear(in_v.y),
                   gamma_2_2_to_linear(in_v.z),
                   in_v.w);
  out[gid] = out_v;
}

/* rgba u8 -> r'g'b'a float */
__kernel void rgbau8_to_rgba_gamma_f (__global const uchar4 * in,
                                      __global       float4 * out)
{
  int gid = get_global_id(0);
  uchar4 in_v  = in[gid];
  float4 out_v;
  out_v = (float4)(rgbau8_to_rgba_gamma_f_lut[in_v.x],
                   rgbau8_to_rgba_gamma_f_lut[in_v.y],
                   rgbau8_to_rgba_gamma_f_lut[in_v.z],
                   babl_u8_to_float[in_v.w]);
  out[gid] = out_v;
}

/* r'g'b'a float -> rgba u8 */
#define RGBA_GAMMA_F_TO_RGBAU8(name, prepare)                       \
__kernel void name (__global const float4 * in,                     \
                    __global       uchar4 * out)                    \
{                                                                   \
  int gid = get_global_id(0);                                       \
  float4 in_v  = prepare (in[gid]);                                 \
  float4 out_v;                                                     \
  out_v = (float4)(gamma_2_2_to_linear(in_v.x),                     \
                   gamma_2_2_to_linear(in_v.y),                     \
                   gamma_2_2_to_linear(in_v.z),                     \
                   in_v.w);                                         \
  out[gid] = float4_to_u8 (out_v);                                  \
}

RGBA_GAMMA_F_TO_RGBAU8 (rgba_gamma_f_to_rgbau8, )
RGBA_GAMMA_F_TO_RGBAU8 (rgba_gamma_f_to_rgbau8_assoc, premultiply_roundtrip)

/* -- Y'CbCrA float -- */

/* RGBA float -> Y'CbCrA float */
__kernel void rgbaf_to_ycbcraf (__global const float4 * in,
                                __global       float4 * out)
{
  int gid = get_global_id(0);
  float4 in_v  = in[gid];
  float4 out_v;

#if (__OPENCL_VERSION__ != CL_VERSION_1_0)
  float3 rgb = (float3)(linear_to_gamma_2_2_pow (in_v.x),
                        linear_to_gamma_2_2_pow (in_v.y),
                        linear_to_gamma_2_2_pow (in_v.z));
#else
  float4 rgb = (float4)(linear_to_gamma_2_2_pow (in_v.x),
                        linear_to_gamma_2_2_pow (in_v.y),
                        linear_to_gamma_2_2_pow (in_v.z), 1.0f);
#endif

  out_v = (float4)(rgb_to_ycbcr (rgb.x, rgb.y, rgb.z), in_v.w);
  out[gid] = out_v;
}

/* Y'CbCrA float -> RGBA float */
__kernel void ycbcraf_to_rgbaf (__global const float4 * in,
                                __global       float4 * out)
{
  int gid = get_global_id(0);
  float4 in_v  = in[gid];
  float4 out_v;

  float4 rgb = (float4)(ycbcr_to_rgb (in_v.x, in_v.y, in_v.z), 0.0f);

  out_v = (float4)(gamma_2_2_to_linear_pow (rgb.x),
                   gamma_2_2_to_linear_pow (rgb.y),
                   gamma_2_2_to_linear_pow (rgb.z),
                   in_v.w);
  out[gid] = out_v;
}

/* RGBA u8 -> Y'CbCrA float */
__kernel void rgbau8_to_ycbcraf (__global const uchar4 * in,
                                 __global       float4 * out)
{
  int gid = get_global_id(0);
  float4 in_v  = u8_to_float4 (in[gid]);
  float4 out_v;

  float4 rgb = (float4)(linear_to_gamma_2_2_pow (in_v.x),
                        linear_to_gamma_2_2_pow (in_v.y),
                        linear_to_gamma_2_2_pow (in_v.z),
                        0.0f);

  out_v = (float4)(rgb_to_ycbcr (rgb.x, rgb.y, rgb.z), in_v.w);
  out[gid] = out_v;
}

/* Y'CbCrA float -> RGBA u8 */
__kernel void ycbcraf_to_rgbau8 (__global const float4 * in,
                                 __global       uchar4 * out)
{
  int gid = get_global_id(0);
  float4 in_v  = in[gid];
  float4 out_v;

  float4 rgb = (float4)(ycbcr_to_rgb (in_v.x, in_v.y, in_v.z), 0.0f);

  out_v = (float4)(gamma_2_2_to_linear_pow (rgb.x),
                   gamma_2_2_to_linear_pow (rgb.y),
                   gamma_2_2_to_linear_pow (rgb.z),
                   in_v.w);
  out[gid] = float4_to_u8 (out_v);
}

/* -- RGB u8 -- */

/* RGB u8 -> RGBA float */
__kernel void rgbu8_to_rgbaf (__global const uchar  * in,
                              __global       float4 * out)
{
  int gid = get_global_id(0);
#if (__OPENCL_VERSION__ != CL_VERSION_1_0)
  uchar3 in_v;
  float4 out_v;
  in_v = vload3 (gid, in);
  out_v.x   = babl_u8_to_float[in_v.x];
  out_v.y   = babl_u8_to_float[in_v.y];
  out_v.z   = babl_u8_to_float[in_v.z];
  out_v.w   = 1.0f;
#else
  float4 out_v = (float4) (babl_u8_to_float[in[3 * gid]],
                           babl_u8_to_float[in[3 * gid + 1]],
                           babl_u8_to_float[in[3 * gid + 2]],
                           1.0f);
#endif
  out[gid]  = out_v;
}

/* RGBA float -> RGB u8 */
__kernel void rgbaf_to_rgbu8 (__global const float4 * in,
                              __global       uchar  * out)
{
  int gid = get_global_id(0);
  float4 in_v  = in[gid];
#if (__OPENCL_VERSION__ != CL_VERSION_1_0)
  uchar3 out_v = float3_to_u8 (in_v.xyz);
  vstore3 (out_v, gid, out);
#else
  uchar4 out_v = float4_to_u8 (in_v);
  out[3 * gid]     = out_v.x;
  out[3 * gid + 1] = out_v.y;
  out[3 * gid + 2] = out_v.z;
#endif
}

/* -- Y u8 -- */

/* Y u8 -> Y float */
__kernel void yu8_to_yf (__global const uchar * in,
                         __global       float * out)
{
  int gid = get_global_id(0);
  float in_v  = babl_u8_to_float[in[gid]];
  float out_v;
  out_v = in_v;
  out[gid] = out_v;
}

/* -- Y float -- */

/* Y float -> RaGaBaA float */
__kernel void yf_to_ragabaf (__global const float * in,
                              __global       float4 * out)
{
  int gid = get_global_id(0);
  float  y  = in[gid];
  float4 out_v = (float4) (y, y, y, 1.0f);

  out[gid] = out_v;
}

/* -- YA float -- */

/* RGBA float -> YA float */
#define RGBAF_TO_YAF(name, luminance)                               \
__kernel void name (__global const float4 * in,                     \
                    __global       float2 * out)                    \
{                                                                   \
  int gid = get_global_id(0);                                       \
  float4 in_v  = in[gid];                                           \
                                                                    \
  out[gid] = (float2) (luminance (in_v.x, in_v.y, in_v.z), in_v.w); \
}

RGBAF_TO_YAF (rgbaf_to_yaf, luminance)
RGBAF_TO_YAF (rgbaf_to_yaf_grey, luminance_grey)

/* RaGaBaA float -> YA float */
#define RAGABAF_TO_YAF(name, luminance)                                \
__kernel void name (__global const float4 * in,                        \
                    __global       float2 * out)                       \
{                                                                      \
  int gid = get_global_id(0);                                          \
  float4 in_v  = in[gid];                                              \
  float4 tmp_v = unpremultiply (in_v);                                 \
                                                                       \
  out[gid] = (float2) (luminance (tmp_v.x, tmp_v.y, tmp_v.z), in_v.w); \
}

RAGABAF_TO_YAF (ragabaf_to_yaf, luminance)
RAGABAF_TO_YAF (ragabaf_to_yaf_grey, luminance_grey)

/* YA float -> RGBA float */
__kernel void yaf_to_rgbaf (__global const float2 * in,
                            __global       float4 * out)
{
  int gid = get_global_id(0);
  float2 in_v  = in[gid];
  float4 out_v = (float4) (in_v.x, in_v.x, in_v.x, in_v.y);

  out[gid] = out_v;
}

/* YA float -> RaGaBaA float */
__kernel void yaf_to_ragabaf (__global const float2 * in,
                              __global       float4 * out)
{
  int gid = get_global_id(0);
  float2 in_v  = in[gid];
  float  y     = in_v.x * babl_epsilon_for_zero (in_v.y);
  float4 out_v = (float4) (y, y, y, in_v.y);

  out[gid] = out_v;
}

/* RGBA u8 -> YA float */
#define RGBAU8_TO_YAF(name, luminance)                              \
__kernel void name (__global const uchar4 * in,                     \
                    __global       float2 * out)                    \
{                                                                   \
  int gid = get_global_id(0);                                       \
  float4 in_v  = u8_to_float4 (in[gid]);                            \
                                                                    \
  out[gid] = (float2) (luminance (in_v.x, in_v.y, in_v.z), in_v.w); \
}

RGBAU8_TO_YAF (rgbau8_to_yaf, luminance)
RGBAU8_TO_YAF (rgbau8_to_yaf_grey, luminance_grey)

/* YA float -> RGBA u8 */
__kernel void yaf_to_rgbau8 (__global const float2 * in,
                             __global       uchar4 * out)
{
  int gid = get_global_id(0);
  float2 in_v  = in[gid];
  float4 out_v = (float4) (in_v.x, in_v.x, in_v.x, in_v.y);

  out[gid] = float4_to_u8 (out_v);
}

/* YA float -> R'G'B'A u8 */
__kernel void yaf_to_rgba_gamma_u8 (__global const float2 * in,
                                    __global       uchar4 * out)
{
  int gid = get_global_id(0);
  float2 in_v  = in[gid];
  float4 tmp_v = (float4) (in_v.x, in_v.x, in_v.x, in_v.y);

  float4 out_v;
  out_v = (float4)(linear_to_gamma_2_2(tmp_v.x),
                   linear_to_gamma_2_2(tmp_v.y),
                   linear_to_gamma_2_2(tmp_v.z),
                   tmp_v.w);
  out[gid] = float4_to_u8 (out_v);
}

/* YA float -> R'G'B' u8 */
__kernel void yaf_to_rgb_gamma_u8 (__global const float2 * in,
                                   __global       uchar  * out)
{
  int gid = get_global_id(0);
  float2 in_v  = in[gid];
  uchar  tmp = float_to_u8 (linear_to_gamma_2_2 (in_v.x));

#if (__OPENCL_VERSION__ != CL_VERSION_1_0)
  vstore3 ((uchar3)tmp, gid, out);
#else
  out[3 * gid] = out[3 * gid + 1] = out[3 * gid + 2] = tmp;
#endif
}

/* R'G'B'A u8 */

/* rgba float -> r'g'b'a u8 */
#define RGBAF_TO_RGBA_GAMMA_U8(name, finish)                        \
__kernel void name (__global const float4 * in,                     \
                    __global       uchar4 * out)                    \
{                                                                   \
  int gid = get_global_id(0);                                       \
  float4 in_v  = in[gid];                                           \
  float4 out_v;                                                     \
  out_v = (float4)(linear_to_gamma_2_2(in_v.x),                     \
                   linear_to_gamma_2_2(in_v.y),                     \
                   linear_to_gamma_2_2(in_v.z),                     \
                   in_v.w);                                         \
  out[gid] = float4_to_u8 (finish (out_v));                         \
}

RGBAF_TO_RGBA_GAMMA_U8 (rgbaf_to_rgba_gamma_u8, )
RGBAF_TO_RGBA_GAMMA_U8 (rgbaf_to_rgba_gamma_u8_assoc, premultiply_roundtrip)

/* R'G'B' u8 */

/* rgba float -> r'g'b' u8 */
__kernel void rgbaf_to_rgb_gamma_u8 (__global const float4 * in,
                                     __global       uchar  * out)
{
  int gid = get_global_id(0);
  float4 in_v  = in[gid];
  float4 tmp_v;
  tmp_v = (float4)(linear_to_gamma_2_2(in_v.x),
                   linear_to_gamma_2_2(in_v.y),
                   linear_to_gamma_2_2(in_v.z),
                   in_v.w);
#if (__OPENCL_VERSION__ != CL_VERSION_1_0)
  uchar3 out_v;
  out_v = float3_to_u8 (tmp_v.xyz);
  vstore3 (out_v, gid, out);
#else
  uchar4 out_v = float4_to_u8 (tmp_v);
  out[3 * gid]     = out_v.x;
  out[3 * gid + 1] = out_v.y;
  out[3 * gid + 2] = out_v.z;
#endif
}
