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
 *
 * Copyright 2013 Carlos Zubieta <czubieta.dev@gmail.com>
 */

/* The CPU code computes with these factors in double precision and
 * rounds the results to float.  To get the same rounding without double
 * support, each factor is split into a float and the float of the
 * remainder, and the products and the quotient are computed with the
 * error of the float operation added back.
 */
#define RED_FACTOR     0x1.06d3a0p-1f  /* (float) 0.5133333 */
#define RED_FACTOR_LO  -0x1.62358p-26f /* 0.5133333 - RED_FACTOR */
#define BLUE_FACTOR    0x1.8bf254p-3f  /* (float) 0.1933333 */
#define BLUE_FACTOR_LO 0x1.1749c8p-29f /* 0.1933333 - BLUE_FACTOR */
#define DIVISOR        0x1.06d3a0p+0f  /* (float) (2.0 * 0.5133333) */
#define DIVISOR_LO     -0x1.62358p-25f /* 2.0 * 0.5133333 - DIVISOR */

/* (float) (a * (hi + lo)) */
float
mul_split (float a,
           float hi,
           float lo)
{
  const float p = a * hi;
  return p + (fma (a, hi, -p) + a * lo);
}

/* (float) (a / (hi + lo)) */
float
div_split (float a,
           float hi,
           float lo)
{
  const float q = a / hi;
  return q + (fma (-q, hi, a) - q * lo) / hi;
}

__kernel void cl_red_eye_removal(__global const float4 *in,
                                 __global       float4 *out,
                                                float adjusted_threshold)
{
  int gid     = get_global_id(0);
  float4 in_v = in[gid];
  float adjusted_red       = mul_split (in_v.x, RED_FACTOR, RED_FACTOR_LO);
  float adjusted_green     = in_v.y;
  float adjusted_blue      = mul_split (in_v.z, BLUE_FACTOR, BLUE_FACTOR_LO);
  float tmp;

  if (adjusted_red >= adjusted_green - adjusted_threshold &&
      adjusted_red >= adjusted_blue  - adjusted_threshold)
    {
      tmp = div_split (adjusted_green + adjusted_blue, DIVISOR, DIVISOR_LO);
      in_v.x = clamp(tmp, 0.0f, 1.0f);
    }
  out[gid]  = in_v;
}
