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

/* The CPU version gets its pixels converted to "CIE LCH(ab) alpha float"
 * by babl, which goes through "RGBA float" and "CIE Lab alpha float".  The
 * conversion to LCH is replicated here operation by operation (including
 * babl's cube root approximation), so the noise is applied to identical
 * LCH values.  This matters for gray pixels, whose hue is decided by
 * rounding noise in the a and b components.
 *
 * Whether the compiler babl was built with contracted its multiply-adds
 * into fused multiply-adds (clang on arm64 does, x86_64 without -mfma and
 * armv7 do not) changes that rounding noise, so both variants are
 * implemented and selected by rgb_to_lab.s9.  The host picks the one that
 * reproduces babl's output for a set of probe colors (cl_noise_cie_lch_probe)
 * and does not use the "RGBA float" kernel if neither does.
 */
#pragma OPENCL FP_CONTRACT OFF

#define LAB_EPSILON         0x1.22354ep-7f  /* 216.0f / 24389.0f */
#define LAB_KAPPA           0x1.c3a5eep+9f  /* 24389.0f / 27.0f  */
#define DEGREES_PER_RADIAN  0x1.ca5dcp+5f   /* 180 / M_PI_F      */
#define RADIANS_PER_DEGREE  0x1.1df46cp-6f  /* M_PI_F / 180      */

/* correctly rounded a / b */
float
lch_div (float a,
         float b)
{
  float q = a / b;

  return fma (fma (-q, b, a), 1.0f / b, q);
}

float
lch_cbrt (float x)
{
  uint  i = as_uint (x);
  float u;

  i = i / 4 + i / 16;
  i = i + i / 16;
  i = i + i / 256;
  i = 0x2a5137a0 + i;
  u = as_float (i);
  u = 0x1.555556p-2f * (2.0f * u + lch_div (x, u * u));
  u = 0x1.555556p-2f * (2.0f * u + lch_div (x, u * u));

  return u;
}

/* a * b + c, with or without the rounding of the product */
float
lch_mad (float a,
         float b,
         float c,
         int   contract)
{
  return contract ? fma (a, b, c) : a * b + c;
}

float
lch_lab_f (float t,
           int   contract)
{
  return t > LAB_EPSILON
           ? lch_cbrt (t)
           : lch_div (lch_mad (t, LAB_KAPPA, 16.0f, contract), 116.0f);
}

/* rgb_to_lab: RGBtoXYZ matrix of the space divided by the D50 white in
 * s0-s8, s9 non-zero if babl's multiply-adds are contracted
 */
float4
lch_rgba_to_lcha (float4  rgba,
                  float16 rgb_to_lab)
{
  int   c  = rgb_to_lab.s9 != 0.0f;
  float xr = lch_mad (rgb_to_lab.s2, rgba.z,
                      lch_mad (rgb_to_lab.s0, rgba.x,
                               rgb_to_lab.s1 * rgba.y, c), c);
  float yr = lch_mad (rgb_to_lab.s5, rgba.z,
                      lch_mad (rgb_to_lab.s3, rgba.x,
                               rgb_to_lab.s4 * rgba.y, c), c);
  float zr = lch_mad (rgb_to_lab.s8, rgba.z,
                      lch_mad (rgb_to_lab.s6, rgba.x,
                               rgb_to_lab.s7 * rgba.y, c), c);
  float fx = lch_lab_f (xr, c);
  float fy = lch_lab_f (yr, c);
  float fz = lch_lab_f (zr, c);
  float L  = lch_mad (fy, 116.0f, -16.0f, c);
  float A  = (fx - fy) * 500.0f;
  float B  = (fy - fz) * 200.0f;
  float C  = sqrt (lch_mad (A, A, B * B, c));
  float H  = atan2 (B, A) * DEGREES_PER_RADIAN;

  if (H < 0.0f)
    H += 360.0f;

  return (float4) (L, C, H, rgba.w);
}

float
lch_lab_finv (float f)
{
  float f3 = f * f * f;

  return f3 > LAB_EPSILON ? f3 : (f * 116.0f - 16.0f) / LAB_KAPPA;
}

/* lab_to_rgb: XYZtoRGB matrix of the space multiplied by the D50 white */
float4
lch_lcha_to_rgba (float4  lcha,
                  float16 lab_to_rgb)
{
  float L  = lcha.x;
  float A  = lcha.y * cos (lcha.z * RADIANS_PER_DEGREE);
  float B  = lcha.y * sin (lcha.z * RADIANS_PER_DEGREE);
  float fy = (L + 16.0f) / 116.0f;
  float fx = fy + A / 500.0f;
  float fz = fy - B / 200.0f;
  float yr = L > LAB_KAPPA * LAB_EPSILON ? fy * fy * fy : L / LAB_KAPPA;
  float xr = lch_lab_finv (fx);
  float zr = lch_lab_finv (fz);

  return (float4) (lab_to_rgb.s0 * xr + lab_to_rgb.s1 * yr + lab_to_rgb.s2 * zr,
                   lab_to_rgb.s3 * xr + lab_to_rgb.s4 * yr + lab_to_rgb.s5 * zr,
                   lab_to_rgb.s6 * xr + lab_to_rgb.s7 * yr + lab_to_rgb.s8 * zr,
                   lcha.w);
}

/* gegl_random_float () converts (rand & 0xffff) / 65535 in double
 * precision; one Newton step with fma gives the same correctly rounded
 * float for all 65536 inputs.
 */
float
lch_random_float (__global const int *random_data,
                  GeglRandom          rand,
                  int                 x,
                  int                 y,
                  int                 n)
{
  const float t = 0.00001525902189669642175f;
  float k = (float) (gegl_cl_random_int (random_data, rand, x, y, 0, n) &
                     0xffff);
  float q = k * t;

  return fma (fma (-q, 65535.0f, k), t, q);
}

float
lch_randomize_value (__global const int *random_data,
                     GeglRandom          rand,
                     float               now,
                     float               min_val,
                     float               max_val,
                     int                 wraps_around,
                     float               rand_max,
                     int                 holdness,
                     int                 x,
                     int                 y,
                     int                 n)
{
  float steps = max_val - min_val + 0.5f;
  float rand_val = lch_random_float (random_data, rand, x, y, n++);
  float new_val;
  float flag;
  int   i;

  for (i = 1; i < holdness; i++)
    {
      float tmp = lch_random_float (random_data, rand, x, y, n++);

      if (tmp < rand_val)
        rand_val = tmp;
    }

  flag = (lch_random_float (random_data, rand, x, y, n) < 0.5f) ? -1.0f : 1.0f;
  new_val = now + flag * fmod (rand_max * rand_val, steps);

  if (new_val < min_val)
    {
      if (wraps_around)
        new_val += steps;
      else
        new_val = min_val;
    }

  if (max_val < new_val)
    {
      if (wraps_around)
        new_val -= steps;
      else
        new_val = max_val;
    }

  return new_val;
}

float4
noise_cie_lch (float4              lcha,
               __global const int *random_data,
               GeglRandom          rand,
               int                 x,
               int                 y,
               int                 whole_region_width,
               int                 holdness,
               float               lightness_distance,
               float               chroma_distance,
               float               hue_distance)
{
  int   n         = (3 * holdness + 4) * (x + whole_region_width * y);
  float lightness = lcha.x;
  float chroma    = lcha.y;
  float hue       = lcha.z;

  if (hue_distance > 0.0f && chroma > 0.0f)
    hue = lch_randomize_value (random_data, rand, hue, 0.0f, 359.0f, 1,
                               hue_distance, holdness, x, y, n);

  n += holdness + 1;
  if (chroma_distance > 0.0f)
    {
      if (chroma == 0.0f)
        hue = lch_random_float (random_data, rand, x, y, n) * 360.0f;

      chroma = lch_randomize_value (random_data, rand, chroma, 0.0f, 100.0f,
                                    0, chroma_distance, holdness,
                                    x, y, n + 1);
    }

  n += holdness + 2;
  if (lightness_distance > 0.0f)
    lightness = lch_randomize_value (random_data, rand, lightness,
                                     0.0f, 100.0f, 0, lightness_distance,
                                     holdness, x, y, n);

  return (float4) (lightness, chroma, hue, lcha.w);
}

#define NOISE_CIE_LCH_ARGS                                    \
  __global const int    *random_data,                         \
        GeglRandom       rand,                                \
        int              roi_x,                               \
        int              roi_y,                               \
        int              roi_width,                           \
        int              whole_region_width,                  \
        int              holdness,                            \
        float            lightness_distance,                  \
        float            chroma_distance,                     \
        float            hue_distance,                        \
        float16          rgb_to_lab,                          \
        float16          lab_to_rgb

#define NOISE_CIE_LCH(lcha)                                   \
  noise_cie_lch ((lcha), random_data, rand,                   \
                 roi_x + gid % roi_width, roi_y + gid / roi_width, \
                 whole_region_width, holdness,                \
                 lightness_distance, chroma_distance, hue_distance)

/* "CIE LCH(ab) alpha float" in and out, converted by babl */
__kernel void cl_noise_cie_lch_lch (__global const float4 *in,
                                    __global       float4 *out,
                                    NOISE_CIE_LCH_ARGS)
{
  int gid = get_global_id (0);

  out[gid] = NOISE_CIE_LCH (in[gid]);
}

/* "RGBA float" in and out */
__kernel void cl_noise_cie_lch (__global const float4 *in,
                                __global       float4 *out,
                                NOISE_CIE_LCH_ARGS)
{
  int    gid  = get_global_id (0);
  float4 lcha = NOISE_CIE_LCH (lch_rgba_to_lcha (in[gid], rgb_to_lab));

  out[gid] = lch_lcha_to_rgba (lcha, lab_to_rgb);
}

/* "R'G'B'A u8" (bpp 4) or "R'G'B' u8" (bpp 3) in, "RGBA float" out.
 *
 * Gray pixels get their LCH values from gray_lch, which the host fills
 * using the very babl fish the CPU version uses: their chroma and hue are
 * pure rounding noise, which depends on the exact 8-bit to linear table
 * babl picked.  Other pixels are linearized with lut.
 */
__kernel void cl_noise_cie_lch_u8 (__global const uchar  *in,
                                   __global       float4 *out,
                                   NOISE_CIE_LCH_ARGS,
                                   __global const float  *lut,
                                   __global const float4 *gray_lch,
                                   int                    bpp)
{
  int                   gid = get_global_id (0);
  __global const uchar *p   = in + gid * bpp;
  float                 a;
  float4                lcha;

  a = bpp == 4 ? lch_div ((float) p[3], 255.0f) : 1.0f;

  if (p[0] == p[1] && p[1] == p[2])
    {
      lcha   = gray_lch[p[0]];
      lcha.w = a;
    }
  else
    {
      lcha = lch_rgba_to_lcha ((float4) (lut[p[0]], lut[p[1]], lut[p[2]], a),
                               rgb_to_lab);
    }

  out[gid] = lch_lcha_to_rgba (NOISE_CIE_LCH (lcha), lab_to_rgb);
}

/* "RGBA float" in, "CIE LCH(ab) alpha float" out, without noise; used by
 * the host to check the conversion against babl
 */
__kernel void cl_noise_cie_lch_probe (__global const float4 *in,
                                      __global       float4 *out,
                                      float16                rgb_to_lab)
{
  int gid = get_global_id (0);

  out[gid] = lch_rgba_to_lcha (in[gid], rgb_to_lab);
}
