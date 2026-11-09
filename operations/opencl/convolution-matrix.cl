/* SPDX-License-Identifier: LGPL-3.0-or-later */

#pragma OPENCL FP_CONTRACT OFF

__kernel void
convolution_matrix (__global   const float4 *in,
                    __global         float4 *out,
                    __constant const float  *matrix,
                               int           matrix_size,
                               float         matrixsum,
                               float         inv_divisor,
                               float         offset,
                               int           alpha_weight,
                               int           channels)
{
  const int x = get_global_id (0);
  const int y = get_global_id (1);
  const int width = get_global_size (0);
  const int stride = width + matrix_size - 1;
  const int src = y * stride + x;
  const int dst = y * width + x;
  float4 sum = 0.0f;
  float alpha_sum = 0.0f;

  for (int j = 0; j < matrix_size; j++)
    for (int i = 0; i < matrix_size; i++)
      {
        const float4 pixel = in[src + j * stride + i];
        float4 val = matrix[j * matrix_size + i] * pixel;

        if (alpha_weight)
          {
            val.xyz *= pixel.w;
            alpha_sum += fabs (val.w);
          }
        sum += val;
      }

  float4 result = sum * inv_divisor + offset;

  if (alpha_weight)
    {
      /* The CPU's componentwise path uses != 0, which also accepts NaN. */
      const int has_weight = channels == 15 ? alpha_sum > 0.0f : alpha_sum != 0.0f;
      result.w = has_weight ? (sum.w * inv_divisor) * matrixsum / alpha_sum + offset
                            : offset;
    }

  if (channels != 15)
    {
      const int radius = matrix_size / 2;
      const float4 center = in[src + radius * (stride + 1)];
      const int4 mask = ((int4) channels & (int4) (1, 2, 4, 8)) != 0;
      result = select (center, result, mask);
    }

  out[dst] = result;
}
