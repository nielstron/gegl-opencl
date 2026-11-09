/* SPDX-License-Identifier: GPL-3.0-or-later */

#pragma OPENCL FP_CONTRACT OFF

typedef struct
{
  int2   anchor;
  float2 fraction;
  float2 step_x;
  float2 step_y;
  int2   samples;
} SamplingGeometry;

static inline float
cubic_weight (float x)
{
  const float a = fabs (x);
  const float a2 = a * a;
  const float a3 = a2 * a;

  /* GEGL's cubic sampler uses the Keys spline with B = 1/2, C = 1/4. */
  if (a < 1.0f)
    return (6.0f * a3 - 10.5f * a2 + 5.0f) * (1.0f / 6.0f);
  if (a <= 2.0f)
    return (-2.0f * a3 + 10.5f * a2 - 18.0f * a + 10.0f) * (1.0f / 6.0f);
  return 0.0f;
}

static inline float4
read_pixel (__global const float4 *src,
            int2                   size,
            int2                   pixel,
            int                    policy,
            float4                 abyss_color)
{
  if (any (pixel < 0) || any (pixel >= size))
    {
      if (policy == 1) /* GEGL_ABYSS_CLAMP */
        pixel = clamp (pixel, (int2) 0, size - 1);
      else if (policy == 2) /* GEGL_ABYSS_LOOP */
        {
          pixel %= size;
          pixel += select ((int2) 0, size, pixel < 0);
        }
      else
        return abyss_color;
    }
  return src[(size_t) pixel.y * size.x + pixel.x];
}

static inline float4
sample_cubic (__global const float4 *src,
              int2                   size,
              int2                   anchor,
              float2                 offset,
              int                    policy,
              float4                 abyss_color)
{
  const int2 whole = convert_int2_rtn (offset);
  const float2 frac = offset - convert_float2 (whole);
  const int2 origin = anchor + whole - 1;
  const float wx[4] = {cubic_weight (frac.x + 1), cubic_weight (frac.x),
                       cubic_weight (frac.x - 1), cubic_weight (frac.x - 2)};
  float4 result = (float4) 0.0f;

  for (int j = 0; j < 4; j++)
    {
      const float wy = cubic_weight (frac.y - (j - 1));
      for (int i = 0; i < 4; i++)
        result += (wx[i] * wy) * read_pixel (src, size, origin + (int2) (i, j),
                                             policy, abyss_color);
    }
  return result;
}

__kernel void
fractal_trace (__global const float4           *src,
               int2                             size,
               int                              policy,
               float4                           abyss_color,
               __global const SamplingGeometry *geometry,
               __global float4                 *dst,
               int                              dst_offset)
{
  const size_t index = get_global_id (0);
  const SamplingGeometry g = geometry[index];
  float4 result = (float4) 0.0f;

  for (int y = 0; y < g.samples.y; y++)
    for (int x = 0; x < g.samples.x; x++)
      {
        const float2 offset = g.fraction + x * g.step_x + y * g.step_y;
        result += sample_cubic (src, size, g.anchor, offset, policy, abyss_color);
      }
  dst[dst_offset + index] = result / (float) (g.samples.x * g.samples.y);
}
