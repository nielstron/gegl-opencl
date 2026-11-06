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
 * Copyright 2013 Victor Oliveira <victormatheus@gmail.com>
 * Copyright 2013 Carlos Zubieta  <czubieta.dev@gmail.com>
 */

/* Block means.  Blocks are numbered from the top-left block of the
 * chunk; block (bx, by) starts at block_x + bx * xsize,
 * block_y + by * ysize and is clipped to bbox (x0, y0, x1, y1).  Blocks
 * that lie completely outside bbox get a zero mean, the fill kernel
 * never uses it.
 */

int
clip_end (int start,
          int size,
          int limit)
{
  /* min (start + size, limit) without overflowing for huge sizes */
  return limit - start > size ? start + size : limit;
}

/* One work-item per block, for small blocks. */
__kernel void pixelize_block_mean (__global const float4 *in,
                                   __global       float4 *blocks,
                                                  int     in_x,
                                                  int     in_y,
                                                  int     in_width,
                                                  int     block_x,
                                                  int     block_y,
                                                  int     xsize,
                                                  int     ysize,
                                                  int4    bbox,
                                                  int     block_count_x)
{
  int bx = get_global_id (0);
  int by = get_global_id (1);

  int x0 = block_x + bx * xsize;
  int y0 = block_y + by * ysize;
  int x1 = clip_end (x0, xsize, bbox.s2);
  int y1 = clip_end (y0, ysize, bbox.s3);
  float4 sum = (float4) (0.0f);
  int i, j;

  x0 = max (x0, bbox.s0);
  y0 = max (y0, bbox.s1);

  if (x1 > x0 && y1 > y0)
    {
      for (j = y0; j < y1; j++)
        {
          __global const float4 *row = in + (j - in_y) * in_width;

          for (i = x0 - in_x; i < x1 - in_x; i++)
            sum += row[i];
        }

      sum /= (float) ((x1 - x0) * (y1 - y0));
    }

  blocks[by * block_count_x + bx] = sum;
}

/* One work-group per block, for large blocks. The local size must be
 * PIXELIZE_GROUP_SIZE.
 */
#define PIXELIZE_GROUP_SIZE 64

__kernel __attribute__((reqd_work_group_size (PIXELIZE_GROUP_SIZE, 1, 1)))
void pixelize_block_mean_wg (__global const float4 *in,
                             __global       float4 *blocks,
                                            int     in_x,
                                            int     in_y,
                                            int     in_width,
                                            int     block_x,
                                            int     block_y,
                                            int     xsize,
                                            int     ysize,
                                            int4    bbox,
                                            int     block_count_x)
{
  __local float4 partial[PIXELIZE_GROUP_SIZE];

  int lid = get_local_id (0);
  int bx  = get_group_id (0);
  int by  = get_global_id (1);

  int x0 = block_x + bx * xsize;
  int y0 = block_y + by * ysize;
  int x1 = clip_end (x0, xsize, bbox.s2);
  int y1 = clip_end (y0, ysize, bbox.s3);
  float4 sum = (float4) (0.0f);
  int width, height;
  int s;

  x0 = max (x0, bbox.s0);
  y0 = max (y0, bbox.s1);
  width  = x1 - x0;
  height = y1 - y0;

  if (width > 0 && height > 0)
    {
      /* walk the block in row-major order, PIXELIZE_GROUP_SIZE pixels
       * apart, so neighbouring work-items read neighbouring pixels
       */
      int step_y = PIXELIZE_GROUP_SIZE / width;
      int step_x = PIXELIZE_GROUP_SIZE - step_y * width;
      int i = lid % width;
      int j = lid / width;

      __global const float4 *base = in + (y0 - in_y) * in_width + (x0 - in_x);

      while (j < height)
        {
          sum += base[j * in_width + i];

          i += step_x;
          j += step_y;
          if (i >= width)
            {
              i -= width;
              j++;
            }
        }
    }

  partial[lid] = sum;
  barrier (CLK_LOCAL_MEM_FENCE);

  for (s = PIXELIZE_GROUP_SIZE / 2; s > 0; s >>= 1)
    {
      if (lid < s)
        partial[lid] += partial[lid + s];
      barrier (CLK_LOCAL_MEM_FENCE);
    }

  if (lid == 0)
    blocks[by * block_count_x + bx] =
      width > 0 && height > 0 ? partial[0] / (float) (width * height)
                              : (float4) (0.0f);
}

/* Fill the output.  For each output column, columns.x is its block
 * column, or -1 when the column is outside the bounding box, and
 * columns.y is its position inside the block.  For each output row,
 * rows.x is its block row, or -1 when the row is outside the bounding
 * box, and pixels whose position inside the block lies in
 * [rows.y, rows.z) are inside the block's shape.  The tables are
 * computed on the host with the same arithmetic as the CPU path.
 */
__kernel void pixelize_fill (__global const float4 *blocks,
                             __global       float4 *out,
                             __global const int2   *columns,
                             __global const int4   *rows,
                                            float4  bg_color,
                                            int     block_count_x,
                                            int     width)
{
  int x = get_global_id (0);
  int y = get_global_id (1);

  int2   column = columns[x];
  int4   row    = rows[y];
  float4 color  = bg_color;

  if (column.x >= 0 && row.x >= 0 &&
      column.y >= row.y && column.y < row.z)
    color = blocks[row.x * block_count_x + column.x];

  out[y * width + x] = color;
}
