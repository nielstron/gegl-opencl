/* SPDX-License-Identifier: LGPL-3.0-or-later */

#pragma OPENCL FP_CONTRACT OFF

/* A pixel belongs to one adjacent pair of blur levels. Its output slot
 * holds the lower sample until the next level completes the blend. This
 * lets every Gaussian pass write to the same blur buffer.
 */
__kernel void
variable_blur_blend (__global const float  *mask,
                     __global const float4 *blur,
                     __global       float4 *out,
                     int                    source_stride,
                     int                    source_offset,
                     float                  previous_edge,
                     float                  edge,
                     float                  next_edge,
                     float                  previous_inv_range,
                     int                    level,
                     int                    levels)
{
  const size_t x = get_global_id (0);
  const size_t y = get_global_id (1);
  const size_t index = y * get_global_size (0) + x;
  const size_t source_index = source_offset + y * source_stride + x;
  const float value = mask[index];
  const int last = levels - 1;

  /* The first and last pairs also cover out-of-range masks.
   * An unordered mask belongs to the first pair and propagates NaN
   * through interpolation, as in piecewise-blend.
   */
  if (level > 0 &&
      (level == 1 || value >= previous_edge) &&
      (level == last || !(value >= edge)))
    {
      const float4 lower = out[index];
      const float4 upper = blur[source_index];
      const float weight = (value - previous_edge) * previous_inv_range;

#if VARIABLE_BLUR_USE_FMA
      out[index] = fma ((float4) weight, upper - lower, lower);
#else
      out[index] = lower + weight * (upper - lower);
#endif
    }
  else if (level < last &&
           (level == 0 || value >= edge) &&
           (level == last - 1 || !(value >= next_edge)))
    {
      out[index] = blur[source_index];
    }
}
