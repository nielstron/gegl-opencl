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

/* The CPU version computes floor (sin (angle) * dist) in double
 * precision.  The angle only depends on the low 16 bits of a random
 * number, so the host uploads a table with sin and cos of all 65536
 * possible angles, each split in a high and a low float.  The product
 * with the (integer) distance is then evaluated with an error-free
 * transformation, which gives the same floor as the double computation.
 */
int
spread_floor_mul (float hi,
                  float lo,
                  int   dist)
{
  float d = (float) dist;
  float p = hi * d;
  float e = fma (hi, d, -p) + lo * d;
  float f = floor (p);

  if (p == f && e < 0.0f)
    f -= 1.0f;

  return (int) f;
}

/* Pixels are moved as opaque blobs of bpp bytes, so any pixel format works
 * and the result is bit-exact.  in is the result rect grown by area_x and
 * area_y pixels on each side (abyss clamped by the buffer iterator).
 */
__kernel void cl_noise_spread (__global const uchar  *in,
                               __global       uchar  *out,
                               __global const int    *random_data,
                               __global const float4 *angle_table,
                                     GeglRandom       rand,
                                     int              roi_x,
                                     int              roi_y,
                                     int              roi_width,
                                     int              in_width,
                                     int              area_x,
                                     int              area_y,
                                     int              bpp)
{
  int    gid = get_global_id (0);
  int    gy  = gid / roi_width;
  int    gx  = gid - gy * roi_width;
  int    x   = roi_x + gx;
  int    y   = roi_y + gy;
  int    xdist = 0;
  int    ydist = 0;
  uint   rnd;
  float4 sc;
  int    src, b;

  if (area_x > 0)
    {
      rnd   = gegl_cl_random_int (random_data, rand, x, y, 0, 0);
      xdist = (int) (rnd % (uint) (2 * area_x + 1)) - area_x;
    }
  if (area_y > 0)
    {
      rnd   = gegl_cl_random_int (random_data, rand, x, y, 0, 1);
      ydist = (int) (rnd % (uint) (2 * area_y + 1)) - area_y;
    }

  rnd = gegl_cl_random_int (random_data, rand, x, y, 0, 2);
  sc  = angle_table[rnd & 0xffff];

  x += spread_floor_mul (sc.x, sc.y, xdist);
  y += spread_floor_mul (sc.z, sc.w, ydist);

  src = (x - roi_x + area_x) + (y - roi_y + area_y) * in_width;

  if ((bpp & 3) == 0)
    {
      __global const uint *s = (__global const uint *) (in + (size_t) src * bpp);
      __global       uint *d = (__global       uint *) (out + (size_t) gid * bpp);

      for (b = 0; b < bpp / 4; b++)
        d[b] = s[b];
    }
  else
    {
      __global const uchar *s = in + (size_t) src * bpp;
      __global       uchar *d = out + (size_t) gid * bpp;

      for (b = 0; b < bpp; b++)
        d[b] = s[b];
    }
}
