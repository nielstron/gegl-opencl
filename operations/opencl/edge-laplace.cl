/* SPDX-License-Identifier: GPL-3.0-or-later */

#pragma OPENCL FP_CONTRACT OFF

static float4
gradient_at (global const float4 *in, int center, int stride)
{
  const float3 a = in[center - stride - 1].xyz;
  const float3 b = in[center - stride].xyz;
  const float3 c = in[center - stride + 1].xyz;
  const float3 d = in[center - 1].xyz;
  const float4 e = in[center];
  const float3 f = in[center + 1].xyz;
  const float3 g = in[center + stride - 1].xyz;
  const float3 h = in[center + stride].xyz;
  const float3 i = in[center + stride + 1].xyz;
  const float3 min_bh = select (b, h, b > h);
  const float3 max_bh = select (h, b, b > h);
  const float3 min_df = select (d, f, d > f);
  const float3 max_df = select (f, d, d > f);
  const float3 minval = fmin (select (min_df, min_bh, min_bh < min_df), e.xyz);
  const float3 maxval = fmax (select (max_df, max_bh, max_bh > max_df), e.xyz);
  const float3 magnitude = 0.5f * fmax (maxval - e.xyz, e.xyz - minval);
  const float3 laplacian = a + b + c + d - 8.0f * e.xyz + f + g + h + i;

  /* The CPU compares the float Laplacian to the double value 1e-5. */
  return (float4) (select (magnitude, -magnitude, laplacian <= 1e-5f), e.w);
}

kernel void
edge_laplace (global const float4 *in,
              global       float4 *out,
              int                  width,
              int                  height,
              local        float4 *gradients)
{
  const int lx = get_local_id (0);
  const int ly = get_local_id (1);
  const int lw = get_local_size (0);
  const int lh = get_local_size (1);
  const int tile_width = lw + 2;
  const int tile_height = lh + 2;
  const int x0 = get_group_id (0) * lw;
  const int y0 = get_group_id (1) * lh;
  const int x = x0 + lx;
  const int y = y0 + ly;
  const int stride = width + 4;

  /* Compute a tile of first-pass gradients plus a one-pixel halo. The
   * source itself has a two-pixel halo around the output rectangle.
   */
  for (int p = ly * lw + lx; p < tile_width * tile_height; p += lw * lh)
    {
      const int sx = min (x0 + p % tile_width, width + 1) + 1;
      const int sy = min (y0 + p / tile_width, height + 1) + 1;
      gradients[p] = gradient_at (in, sy * stride + sx, stride);
    }

  barrier (CLK_LOCAL_MEM_FENCE);

  if (x < width && y < height)
    {
      local const float4 *p = gradients + (ly + 1) * tile_width + lx + 1;
      const float4 center = *p;
      const int3 crosses_zero =
        (p[-tile_width - 1].xyz < 0.0f) | (p[-tile_width].xyz < 0.0f) |
        (p[-tile_width + 1].xyz < 0.0f) | (p[-1].xyz < 0.0f) |
        (p[1].xyz < 0.0f) | (p[tile_width - 1].xyz < 0.0f) |
        (p[tile_width].xyz < 0.0f) | (p[tile_width + 1].xyz < 0.0f);

      out[y * width + x] = (float4) (select ((float3) 0.0f, center.xyz,
                                             (center.xyz > 0.0f) & crosses_zero),
                                     center.w);
    }
}
