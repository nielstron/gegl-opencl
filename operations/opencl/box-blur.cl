/* Separable box blur.
 *
 * Both passes use a sliding window: every work-item produces a run of
 * consecutive output pixels, so the cost per pixel is independent of the
 * radius.  The input of the horizontal pass has the full radius margin on
 * all sides; it writes (height + 2 * radius) rows of width pixels into aux,
 * which the vertical pass then reduces to height rows.
 */

__kernel void kernel_blur_hor (__global const float4 *in,
                               __global       float4 *aux,
                               int width, int rows, int radius, int run)
{
  const int x0  = get_global_id (0) * run;
  const int row = get_global_id (1);

  if (row >= rows || x0 >= width)
    return;

  {
    const int   in_width = width + 2 * radius;
    const float rad1     = 1.0f / (float)(2 * radius + 1);
    const int   x1       = min (x0 + run, width);
    __global const float4 *src = in + row * in_width;
    __global       float4 *dst = aux + row * width;
    float4 sum = (float4)(0.0f);
    int    i;

    for (i = 0; i <= 2 * radius; i++)
      sum += src[x0 + i];

    dst[x0] = sum * rad1;

    for (i = x0 + 1; i < x1; i++)
      {
        sum += src[i + 2 * radius] - src[i - 1];
        dst[i] = sum * rad1;
      }
  }
}

__kernel void kernel_blur_ver (__global const float4 *aux,
                               __global       float4 *out,
                               int width, int height, int radius, int run)
{
  const int x  = get_global_id (0);
  const int y0 = get_global_id (1) * run;

  if (x >= width || y0 >= height)
    return;

  {
    const float rad1 = 1.0f / (float)(2 * radius + 1);
    const int   y1   = min (y0 + run, height);
    __global const float4 *src = aux + x;
    __global       float4 *dst = out + x;
    float4 sum = (float4)(0.0f);
    int    i;

    for (i = y0; i <= y0 + 2 * radius; i++)
      sum += src[i * width];

    dst[y0 * width] = sum * rad1;

    for (i = y0 + 1; i < y1; i++)
      {
        sum += src[(i + 2 * radius) * width] - src[(i - 1) * width];
        dst[i * width] = sum * rad1;
      }
  }
}
