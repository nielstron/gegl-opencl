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
 */

/* Conversions between 8-bit and float formats that give the same result
 * as babl, for conversions that work on each channel separately.
 *
 * Instead of evaluating transfer functions and rounding on the device,
 * which depends on the precision of the device's math functions and on
 * the rounding of ties, the kernels use tables computed on the host from
 * babl itself.
 *
 * From float to 8-bit, every channel is quantized with a table of
 * thresholds: lut[k] (k = 1..255) is the smallest float that babl maps to
 * k or more, and the 8-bit value is found with a binary search of eight
 * comparisons.  From 8-bit to float, lut[k] is babl's float value for k.
 *
 * Each kernel takes a table of 512 floats: the table for the color
 * channels followed by the table for the alpha channel.  The kernels are
 * shared by several pairs of formats, with different tables.
 */

/* babl/babl-internal.h */
#define BABL_ALPHA_FLOOR_F (1.0f / 65536.0f)

uchar  quantize      (float                 value,
                      __global const float *lut);
uchar4 quantize_rgba (float4                value,
                      __global const float *lut);

uchar
quantize (float                 value,
          __global const float *lut)
{
  int i = 0;

  /* comparisons with NaN are false, giving 0 */
  i += (lut[i + 128] <= value) ? 128 : 0;
  i += (lut[i +  64] <= value) ?  64 : 0;
  i += (lut[i +  32] <= value) ?  32 : 0;
  i += (lut[i +  16] <= value) ?  16 : 0;
  i += (lut[i +   8] <= value) ?   8 : 0;
  i += (lut[i +   4] <= value) ?   4 : 0;
  i += (lut[i +   2] <= value) ?   2 : 0;
  i += (lut[i +   1] <= value) ?   1 : 0;

  return (uchar) i;
}

uchar4
quantize_rgba (float4                value,
               __global const float *lut)
{
  return (uchar4) (quantize (value.x, lut),
                   quantize (value.y, lut),
                   quantize (value.z, lut),
                   quantize (value.w, lut + 256));
}

/* RGBA float -> RGBA u8, per-channel conversions */
__kernel void quantize_rgbaf_to_u8 (__global const float4 * in,
                                    __global       uchar4 * out,
                                    __global const float  * lut)
{
  int gid = get_global_id (0);

  out[gid] = quantize_rgba (in[gid], lut);
}

/* RaGaBaA float -> RGBA u8, un-premultiplying first */
__kernel void quantize_ragabaf_to_u8 (__global const float4 * in,
                                      __global       uchar4 * out,
                                      __global const float  * lut)
{
  int    gid   = get_global_id (0);
  float4 in_v  = in[gid];
  float  alpha = in_v.w;
  float  recip;

  /* un-premultiply like babl does */
  if (alpha < BABL_ALPHA_FLOOR_F)
    {
      if (alpha >= 0.0f)
        alpha = BABL_ALPHA_FLOOR_F;
      else if (alpha >= -BABL_ALPHA_FLOOR_F)
        alpha = -BABL_ALPHA_FLOOR_F;
    }
  recip = 1.0f / alpha;

  in_v.xyz *= recip;

  out[gid] = quantize_rgba (in_v, lut);
}

/* YA float -> RGBA u8 */
__kernel void quantize_yaf_to_u8 (__global const float2 * in,
                                  __global       uchar4 * out,
                                  __global const float  * lut)
{
  int    gid  = get_global_id (0);
  float2 in_v = in[gid];
  uchar  y    = quantize (in_v.x, lut);

  out[gid] = (uchar4) (y, y, y, quantize (in_v.y, lut + 256));
}

/* RGBA u8 -> RGBA float, per-channel conversions */
__kernel void expand_rgbau8_to_f (__global const uchar4 * in,
                                  __global       float4 * out,
                                  __global const float  * lut)
{
  int    gid  = get_global_id (0);
  uchar4 in_v = in[gid];

  out[gid] = (float4) (lut[in_v.x],
                       lut[in_v.y],
                       lut[in_v.z],
                       lut[256 + in_v.w]);
}
