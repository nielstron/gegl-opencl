/* SPDX-License-Identifier: GPL-3.0-or-later */

#pragma OPENCL FP_CONTRACT OFF

/* Matches GeglEdgeAlgo. */
enum
{
  SOBEL,
  PREWITT,
  GRADIENT,
  ROBERTS,
  DIFFERENTIAL,
  LAPLACE
};

static float4
read_pixel (__global const float *in, int index, int has_alpha)
{
  return has_alpha ? vload4 (index, in) : (float4) (vload3 (index, in), 1.0f);
}

static float3
edge_prewitt (float3 a, float3 b, float3 c,
              float3 d, float3 e, float3 f,
              float3 g, float3 h, float3 i)
{
  float3 m = 0.0f;
  e *= 2.0f;

  m = fmax (m,  a + b + c + d - e + f - g - h - i);
  m = fmax (m,  a + b + c + d - e - f + g - h - i);
  m = fmax (m,  a + b - c + d - e - f + g + h - i);
  m = fmax (m,  a - b - c + d - e - f + g + h + i);
  m = fmax (m, -a - b - c + d - e + f + g + h + i);
  m = fmax (m, -a - b + c - d - e + f + g + h + i);
  m = fmax (m, -a + b + c - d - e + f - g + h + i);
  m = fmax (m,  a + b + c - d - e + f - g - h + i);
  return m;
}

__kernel void
edge_detect (__global const float *in,
             __global       float *out,
             int                   algorithm,
             float                 amount,
             int                   has_alpha)
{
  const int x = get_global_id (0);
  const int y = get_global_id (1);
  const int width = get_global_size (0);
  const int stride = width + 2;
  const int center = (y + 1) * stride + x + 1;
  const int dst = y * width + x;
  const float3 a = read_pixel (in, center - stride - 1, has_alpha).xyz;
  const float3 b = read_pixel (in, center - stride,     has_alpha).xyz;
  const float3 c = read_pixel (in, center - stride + 1, has_alpha).xyz;
  const float3 d = read_pixel (in, center - 1,          has_alpha).xyz;
  const float4 e = read_pixel (in, center,              has_alpha);
  const float3 f = read_pixel (in, center + 1,          has_alpha).xyz;
  const float3 g = read_pixel (in, center + stride - 1, has_alpha).xyz;
  const float3 h = read_pixel (in, center + stride,     has_alpha).xyz;
  const float3 i = read_pixel (in, center + stride + 1, has_alpha).xyz;
  float3 dx;
  float3 dy;
  float3 result;

  switch (algorithm)
    {
    case PREWITT:
      result = amount * edge_prewitt (a, b, c, d, e.xyz, f, g, h, i);
      break;
    case LAPLACE:
      result = amount * (a + b + c + d - 8.0f * e.xyz + f + g + h + i);
      break;
    case GRADIENT:
      dx = 4.0f * e.xyz - 4.0f * f;
      dy = -4.0f * e.xyz + 4.0f * h;
      result = sqrt (dx * dx * amount + dy * dy * amount);
      break;
    case ROBERTS:
      dx = 4.0f * e.xyz - 4.0f * i;
      dy = 4.0f * f - 4.0f * h;
      result = sqrt (dx * dx * amount + dy * dy * amount);
      break;
    case DIFFERENTIAL:
      dx = 2.0f * e.xyz - 2.0f * f + 2.0f * h - 2.0f * i;
      dy = -2.0f * e.xyz - 2.0f * f + 2.0f * h + 2.0f * i;
      result = sqrt (dx * dx * amount + dy * dy * amount);
      break;
    default:
    case SOBEL:
      dx = -a + c - 2.0f * d + 2.0f * f - g + i;
      dy = -a - 2.0f * b - c + g + 2.0f * h + i;
      result = sqrt (dx * dx * amount + dy * dy * amount);
      break;
    }

  if (has_alpha)
    vstore4 ((float4) (result, e.w), dst, out);
  else
    vstore3 (result, dst, out);
}
