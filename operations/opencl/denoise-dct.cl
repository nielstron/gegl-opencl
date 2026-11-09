/* This file is an image processing operation for GEGL
 *
 * GEGL is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 */

#pragma OPENCL FP_CONTRACT OFF

/* The forward transforms run in CPU order so that rounding before the hard
 * threshold agrees with the CPU. On targets that use FMA, the CPU contracts
 * each multiply-add into a single instruction.
 */
#if GEGL_DCT_FUSED
#define DCT_STEP(value, weight, sum) fma (value, (float3) (weight), sum)
#else
#define DCT_STEP(value, weight, sum) ((sum) + (value) * (weight))
#endif

/* All patches with the same left edge share their horizontal transform. */
__kernel void
denoise_dct_forward_rows (__global const float4 *input,
                          __global       float4 *rows,
                          __constant const float *basis,
                          int input_stride,
                          int input_offset,
                          int patch_size)
{
  const int coeff = get_global_id (0);
  const int y = get_global_id (1);
  const int origin = coeff / patch_size;
  const int u = coeff % patch_size;
  const int start = input_offset + y * input_stride + origin;
  float3 sum = (float3) 0.0f;

  for (int x = 0; x < patch_size; x++)
    sum = DCT_STEP (input[start + x].xyz, basis[u * patch_size + x], sum);

  rows[y * get_global_size (0) + coeff] = (float4) (sum, 0.0f);
}

/* For each horizontal coefficient, transform and threshold the vertical
 * patches that overlap this band. The inverse DCT is linear, so we can sum
 * the contributions of all such patches before doing the inverse rows.
 * Keeping the band in local memory avoids writing a full patch per pixel.
 */
#define BAND_HEIGHT 8

__kernel void
denoise_dct_columns (__global const float4 *rows,
                     __global       float4 *columns,
                     __constant const float *basis,
                     __local float4 *row_band,
                     __local float4 *coefficients,
                     __local float4 *partial,
                     int source_height,
                     int output_y,
                     int output_height,
                     int patch_size,
                     float threshold)
{
  const int coeff = get_global_id (0);
  const int coeff_stride = get_global_size (0);
  const int lane = get_local_id (0);
  const int v = get_local_id (1);
  const int group_width = get_local_size (0);
  const int band_y = get_group_id (1) * BAND_HEIGHT;
  const int first_y = output_y + band_y - (patch_size - 1);
  const int patch_rows = BAND_HEIGHT + patch_size - 1;
  const int source_rows = patch_rows + patch_size - 1;
  const int coeff_row_stride = patch_size * group_width;

  for (int row = v; row < source_rows; row += patch_size)
    {
      const int y = first_y + row;
      row_band[row * group_width + lane] =
        y >= 0 && y < source_height ?
        rows[y * coeff_stride + coeff] : (float4) 0.0f;
    }

  barrier (CLK_LOCAL_MEM_FENCE);

  for (int patch = 0; patch < patch_rows; patch++)
    {
      const int origin = first_y + patch;
      float3 sum = (float3) 0.0f;

      if (origin >= 0 && origin + patch_size <= source_height)
        {
          for (int y = 0; y < patch_size; y++)
            sum = DCT_STEP (row_band[(patch + y) * group_width + lane].xyz,
                            basis[v * patch_size + y], sum);

          sum = select (sum, (float3) 0.0f, fabs (sum) < threshold);
        }

      coefficients[patch * coeff_row_stride + v * group_width + lane] =
        (float4) (sum, 0.0f);
    }

  barrier (CLK_LOCAL_MEM_FENCE);

  for (int y = 0; y < BAND_HEIGHT; y++)
    {
      float3 sum = (float3) 0.0f;

      for (int rel_y = 0; rel_y < patch_size; rel_y++)
        {
          const int patch = y + patch_size - 1 - rel_y;
          const int index = patch * coeff_row_stride + v * group_width + lane;
          sum = fma (coefficients[index].xyz,
                     (float3) basis[v * patch_size + rel_y], sum);
        }

      partial[y * coeff_row_stride + v * group_width + lane] =
        (float4) (sum, 0.0f);
    }

  barrier (CLK_LOCAL_MEM_FENCE);

  if (v < BAND_HEIGHT && band_y + v < output_height)
    {
      float4 sum = (float4) 0.0f;

      for (int freq = 0; freq < patch_size; freq++)
        sum += partial[v * coeff_row_stride + freq * group_width + lane];

      columns[(band_y + v) * coeff_stride + coeff] = sum;
    }
}

__kernel void
denoise_dct_inverse_rows (__global const float4 *columns,
                          __global       float4 *rows,
                          __constant const float *basis,
                          int patch_size)
{
  const int coeff = get_global_id (0);
  const int y = get_global_id (1);
  const int x = coeff % patch_size;
  const int start = y * get_global_size (0) + coeff - x;
  float3 sum = (float3) 0.0f;

  for (int u = 0; u < patch_size; u++)
    sum = fma (columns[start + u].xyz,
               (float3) basis[u * patch_size + x], sum);

  rows[start + x] = (float4) (sum, 0.0f);
}

/* Count the patches that cover each pixel, excluding patches that would
 * extend beyond the image. Keep the original alpha channel.
 */
__kernel void
denoise_dct_average (__global const float4 *input,
                     __global const float4 *rows,
                     __global       float4 *output,
                     int input_stride,
                     int input_offset,
                     int source_width,
                     int source_height,
                     int output_x,
                     int output_y,
                     int patch_size)
{
  const int x = get_global_id (0) + output_x;
  const int y = get_global_id (1) + output_y;
  const int nx = source_width - patch_size + 1;
  const int ny = source_height - patch_size + 1;
  const int first_x = max (0, x - patch_size + 1);
  const int last_x = min (x, nx - 1);
  const int count_x = last_x - first_x + 1;
  const int count_y = min (y, ny - 1) - max (0, y - patch_size + 1) + 1;
  const int base = (y - output_y) * nx * patch_size + x;
  float3 sum = (float3) 0.0f;

  for (int origin = first_x; origin <= last_x; origin++)
    sum += rows[base + origin * (patch_size - 1)].xyz;

  const float alpha = input[input_offset + y * input_stride + x].w;
  const float3 rgb = sum * (1.0f / (float) (count_x * count_y));
  output[get_global_id (1) * get_global_size (0) + get_global_id (0)] =
    (float4) (rgb, alpha);
}
