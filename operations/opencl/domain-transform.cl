/* This file is an image processing operation for GEGL
 *
 * GEGL is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 */

static inline ushort
edge_difference (uchar3 pixel, uchar3 neighbor)
{
  uchar3 diff = abs_diff (pixel, neighbor);

  return diff.x + diff.y + diff.z;
}

/* The edge weights for every pass refer to the original image, with the
 * same quantized RGB components as the CPU path.
 */
__kernel void
domain_transform_edges (__global const uchar   *source,
                        __global       ushort2 *edges,
                        int width,
                        int source_stride)
{
  int x = get_global_id (0);
  int y = get_global_id (1);
  size_t index = y * width + x;
  __global const uchar *pixel = source + index * source_stride;
  uchar3 center = vload3 (0, pixel);
  ushort horizontal = x > 0 ?
    edge_difference (center, vload3 (0, pixel - source_stride)) : 0;
  ushort vertical = y > 0 ?
    edge_difference (center, vload3 (0, pixel - (size_t) width * source_stride)) : 0;

  edges[index] = (ushort2) (horizontal, vertical);
}

/* Lines are independent. A work item applies both recursive passes to one
 * entire row or column, keeping the feedback order of the CPU algorithm.
 */
__kernel void
domain_transform_filter (__global       float4  *pixels,
                         __global const ushort2 *edges,
                         __constant const float *weights,
                         int table_offset,
                         int width,
                         int height,
                         int vertical)
{
  int line = get_global_id (0);
  int length = vertical ? height : width;
  int stride = vertical ? width : 1;
  int start = vertical ? line : line * width;
  float4 last = pixels[start];
  weights += table_offset;

  for (int k = 0, index = start; k < length; k++, index += stride)
    {
      ushort2 edge = edges[index];
      float w = weights[vertical ? edge.y : edge.x];

      last = (1.0f - w) * pixels[index] + w * last;
      pixels[index] = last;
    }

  int index = start + (length - 1) * stride;
  last = pixels[index];

  for (int k = length - 1; k >= 0; k--, index -= stride)
    {
      int neighbor = k == length - 1 ? index : index + stride;
      ushort2 edge = edges[neighbor];
      float w = weights[vertical ? edge.y : edge.x];

      last = (1.0f - w) * pixels[index] + w * last;
      pixels[index] = last;
    }
}
