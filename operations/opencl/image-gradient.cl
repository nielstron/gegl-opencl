/* SPDX-License-Identifier: LGPL-3.0-or-later */

static float3
read_rgb (__global const void *in, int index, int byte_components)
{
  if (byte_components == 4)
    return convert_float3 (vload4 (index, (__global const uchar *) in).xyz) * (1.0f / 255.0f);
  if (byte_components == 3)
    return convert_float3 (vload3 (index, (__global const uchar *) in)) * (1.0f / 255.0f);
  return vload3 (index, (__global const float *) in);
}

/* Central differences on gamma RGB pixels, with a one-pixel halo. */
__kernel void
image_gradient_magnitude (__global const void  *in,
                          __global       float *out,
                          int                   byte_components,
                          int                   output_ya)
{
  const int x = get_global_id (0);
  const int y = get_global_id (1);
  const int width = get_global_size (0);
  const int stride = width + 2;
  const int center = (y + 1) * stride + x + 1;
  const float3 dx = read_rgb (in, center - 1, byte_components) -
                    read_rgb (in, center + 1, byte_components);
  const float3 dy = read_rgb (in, center - stride, byte_components) -
                    read_rgb (in, center + stride, byte_components);
  const float3 squared = dx * dx + dy * dy;
  float largest = squared.x <= squared.y ? squared.y : squared.x;

  if (largest < squared.z)
    largest = squared.z;

  const float magnitude = sqrt (largest);

  if (output_ya)
    vstore2 ((float2) (magnitude, 1.0f), y * width + x, out);
  else
    out[y * width + x] = magnitude;
}
