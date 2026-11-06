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

/* gegl_random_float () converts (rand & 0xffff) / 65535 in double
 * precision; one Newton step with fma gives the same correctly rounded
 * float for all 65536 inputs.
 */
#pragma OPENCL FP_CONTRACT OFF

uint
noise_rgb_random_k (__global const int *random_data,
                    GeglRandom          rand,
                    int                 x,
                    int                 y,
                    int                 n)
{
  return gegl_cl_random_int (random_data, rand, x, y, 0, n) & 0xffff;
}

float
noise_rgb_k_to_float (uint k)
{
  const float t  = 0.00001525902189669642175f;
  float       kf = (float) k;
  float       q  = kf * t;

  return fma (fma (-q, 65535.0f, kf), t, q);
}

/* Ratio method (Kinderman-Monahan), as in the CPU version.
 *
 * To take the same branches as the CPU, -4 * logf (u) comes from a table
 * computed on the host (log_table, indexed like u by the 16 random bits)
 * and x = sqrt (8 / e) * (v - 0.5) / u, which the CPU evaluates in double
 * precision, is computed with the constant split in a high and a low part
 * and a corrected division.
 */
float
noise_rgb_gauss (__global const int   *random_data,
                 __global const float *log_table,
                 GeglRandom            rand,
                 int                   xx,
                 int                   yy,
                 int                  *n)
{
  const float c_hi = 0x1.b72cd4p+0f;
  const float c_lo = -0x1.99d8dp-29f;
  float x;
  uint  ku;

  do
    {
      float v, u, dv, p, lo, r;

      v = noise_rgb_k_to_float (noise_rgb_random_k (random_data, rand,
                                                    xx, yy, (*n)++));

      do
        ku = noise_rgb_random_k (random_data, rand, xx, yy, (*n)++);
      while (ku == 0);

      u  = noise_rgb_k_to_float (ku);
      dv = v - 0.5f;
      p  = c_hi * dv;
      lo = fma (c_lo, dv, fma (c_hi, dv, -p));
      x  = p / u;
      r  = fma (-x, u, p);
      x  = x + (r + lo) / u;
    }
  while (x * x > log_table[ku]);

  return x;
}

__kernel void cl_noise_rgb (__global const float4 *in,
                            __global       float4 *out,
                            __global const int    *random_data,
                            __global const float  *log_table,
                                  GeglRandom       rand,
                                  int              roi_x,
                                  int              roi_y,
                                  int              roi_width,
                                  float4           amount,
                                  int              correlated,
                                  int              independent,
                                  int              gaussian)
{
  int    gid = get_global_id (0);
  int    gy  = gid / roi_width;
  int    x   = roi_x + gid - gy * roi_width;
  int    y   = roi_y + gy;
  int    n   = 0;
  float  in_v[4];
  float  out_v[4];
  float  noise_coeff = 0.0f;
  float  amt[4] = { amount.x, amount.y, amount.z, amount.w };
  float4 src = in[gid];
  int    b;

  in_v[0] = src.x;
  in_v[1] = src.y;
  in_v[2] = src.z;
  in_v[3] = src.w;

  for (b = 0; b < 4; b++)
    {
      if (b == 0 || independent || b == 3)
        {
          float r;

          if (gaussian)
            r = noise_rgb_gauss (random_data, log_table, rand, x, y, &n);
          else
            r = noise_rgb_k_to_float (noise_rgb_random_k (random_data, rand,
                                                          x, y, n++))
                * 2.0f - 1.0f;

          noise_coeff = amt[b] * r * 0.5f;
        }

      if (noise_coeff != 0.0f)
        {
          float tmp;

          if (correlated)
            tmp = in_v[b] + in_v[b] * (noise_coeff / 0.5f);
          else
            tmp = in_v[b] + noise_coeff;

          out_v[b] = tmp > 1.0f ? 1.0f : (tmp < 0.0f ? 0.0f : tmp);
        }
      else
        {
          out_v[b] = in_v[b];
        }
    }

  out[gid] = (float4) (out_v[0], out_v[1], out_v[2], out_v[3]);
}
