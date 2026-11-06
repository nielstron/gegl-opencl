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

/* Each work item performs the random walk of one output pixel, exactly
 * like the CPU version, and copies the source pixel it ends on.  Pixels
 * are moved as opaque blobs of bpp bytes, so any pixel format works and
 * the result is bit-exact.
 *
 * in is the result rect grown by 'area' pixels on each side (abyss
 * clamped by the buffer iterator), the walk can not leave it.
 *
 * max_k: the CPU compares (float) ((rand & 0xffff) * (100.0 / 65535)) with
 * pct_random, which is monotonic in (rand & 0xffff); the host computes the
 * largest value that passes, or -1 if none does.
 */
__kernel void cl_noise_pick (__global const uchar *in,
                             __global       uchar *out,
                             __global const int   *random_data,
                                   GeglRandom      rand,
                                   int             roi_x,
                                   int             roi_y,
                                   int             roi_width,
                                   int             in_width,
                                   int             area,
                                   int             repeat,
                                   int             max_k,
                                   int             bpp)
{
  int gid = get_global_id (0);
  int gy  = gid / roi_width;
  int gx  = gid - gy * roi_width;
  int pos_x = roi_x + gx;
  int pos_y = roi_y + gy;
  int r, b;
  int src;

  for (r = 0; r < repeat; r++)
    {
      uint rnd = gegl_cl_random_int (random_data, rand, pos_x, pos_y, 0, r);

      if ((int) (rnd & 0xffff) <= max_k)
        {
          int rand2 = (int) (rnd % 9u);

          pos_x += (rand2 % 3) - 1;
          pos_y += (rand2 / 3) - 1;
        }
    }

  src = (pos_x - roi_x + area) + (pos_y - roi_y + area) * in_width;

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
