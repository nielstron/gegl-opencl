float4
get_pixel_color (__global const float4 *in,
                          const int     rect_width,
                          const int     rect_height,
                          const int     rect_x,
                          const int     rect_y,
                          const int     x,
                          const int     y)
{
  const int ix = clamp (x - rect_x, 0, rect_width - 1);
  const int iy = clamp (y - rect_y, 0, rect_height - 1);
  return in[iy * rect_width + ix];
}

float4
bilinear_mix (const float4 p00,
              const float4 p01,
              const float4 p10,
              const float4 p11,
              const float dx,
              const float dy)
{
  return mix (mix (p00, p10, dy), mix (p01, p11, dy), dx);
}

/* Double-float helpers: a value is represented as hi + lo.  They are used
 * to compute the number of samples exactly like the CPU code does in
 * double precision, so that ceil () gives the same result.
 */
float2
df_two_sum (float a,
            float b)
{
  const float s  = a + b;
  const float bb = s - a;
  return (float2)(s, (a - (s - bb)) + (b - bb));
}

float2
df_two_prod (float a,
             float b)
{
  const float p = a * b;
  return (float2)(p, fma (a, b, -p));
}

float2
df_add (float2 a,
        float2 b)
{
  float2 s = df_two_sum (a.x, b.x);
  s.y += a.y + b.y;
  return df_two_sum (s.x, s.y);
}

float2
df_mul (float2 a,
        float2 b)
{
  float2 p = df_two_prod (a.x, b.x);
  p.y += a.x * b.y + a.y * b.x;
  return df_two_sum (p.x, p.y);
}

/* returns TRUE if a < b */
int
df_less (float2 a,
         float2 b)
{
  const float2 d = df_add (a, -b);
  return d.x < 0.0f || (d.x == 0.0f && d.y < 0.0f);
}

/* ceil (sqrt (sq)) for sq given as a double-float */
int
df_ceil_sqrt (float2 sq)
{
  int m = (int) ceil (sqrt (sq.x + sq.y));
  if (m > 0 && ! df_less (df_two_prod ((float) (m - 1), (float) (m - 1)), sq))
    m--;
  else if (df_less (df_two_prod ((float) m, (float) m), sq))
    m++;
  return m;
}

#define NOMINAL_NUM_IT 100

kernel void
cl_motion_blur_circular (__global const float4 *in,
                         __global       float4 *out,
                                  const int     src_width,
                                  const int     src_height,
                                  const int     src_x,
                                  const int     src_y,
                                  const int     dst_x,
                                  const int     dst_y,
                                  const int     whole_region_width,
                                  const int     whole_region_height,
                                  const int     whole_region_x,
                                  const int     whole_region_y,
                                  const float   angle,
                                  const float2  arc_factor_sq,
                                  const float2  center_x,
                                  const float2  center_y)
{
  const int gidx = get_global_id(0);
  const int gidy = get_global_id(1);

  const int x = gidx + dst_x;
  const int y = gidy + dst_y;

  const float2 xr2 = df_add ((float2)((float) x, 0.0f), -center_x);
  const float2 yr2 = df_add ((float2)((float) y, 0.0f), -center_y);

  /* n = ceil (radius * angle * SQRT_2), computed as the ceil of the
   * square root of (xr^2 + yr^2) * (angle * SQRT_2)^2 in double-float
   * precision to match the CPU.  This is not the "real" arc length, a
   * bit shorter.
   */
  const float2 radius_sq = df_add (df_mul (xr2, xr2), df_mul (yr2, yr2));
  int n = df_ceil_sqrt (df_mul (radius_sq, arc_factor_sq));

  /* ensure quality with small angles */
  n = max (n, 3);

  /* performance concern */
  if (n > NOMINAL_NUM_IT)
    {
      const int v = n - NOMINAL_NUM_IT;
      int r = (int) sqrt ((float) v);
      if ((r + 1) * (r + 1) <= v)
        r++;
      else if (r * r > v)
        r--;
      n = NOMINAL_NUM_IT + r;
    }

  float4 sum = (float4)0.0f;
  int count = 0;

  /* Iterate over the arc, rotating the offset of the pixel from the
   * center by theta in [angle / 2, -angle / 2), like the CPU code.
   * The middle sample (theta == 0) lands exactly on the pixel.
   */
  for (int i = 0; i < n; i++)
    {
      const float theta = angle * (float)(n - 2 * i) / (float)(2 * n);
      const float s_val = sin (theta);
      const float h_val = sin (theta * 0.5f);
      const float c_m1  = -2.0f * h_val * h_val; /* cos (theta) - 1 */

      const float fx = (float)x + ((xr2.x * c_m1 - yr2.x * s_val) +
                                   (xr2.y * c_m1 - yr2.y * s_val));
      const float fy = (float)y + ((xr2.x * s_val + yr2.x * c_m1) +
                                   (xr2.y * s_val + yr2.y * c_m1));
      const int ix = (int)floor (fx);
      const int iy = (int)floor (fy);

      if (ix >= whole_region_x && ix < whole_region_x + whole_region_width &&
          iy >= whole_region_y && iy < whole_region_y + whole_region_height)
        {
          /* do bilinear interpolation to get a nice smooth result */
          const float4 p00 = get_pixel_color (
            in, src_width, src_height, src_x, src_y, ix, iy);
          const float4 p01 = get_pixel_color (
            in, src_width, src_height, src_x, src_y, ix + 1, iy);
          const float4 p10 = get_pixel_color (
            in, src_width, src_height, src_x, src_y, ix, iy + 1);
          const float4 p11 = get_pixel_color (
            in, src_width, src_height, src_x, src_y, ix + 1, iy + 1);
          sum += bilinear_mix(p00, p01, p10, p11, fx - ix, fy - iy);
          count++;
        }
    }

  float4 out_v;
  if (count == 0)
    out_v = get_pixel_color (in, src_width, src_height, src_x, src_y, x, y);
  else
    out_v = sum / (float) count;
  out[gidy * get_global_size(0) + gidx] = out_v;
}
