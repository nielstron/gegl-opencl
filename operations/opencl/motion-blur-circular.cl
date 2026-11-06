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

/* Double-float helpers: a value is represented as hi + lo.  They are
 * used to compute the number of samples and the sample positions like
 * the CPU code does in double precision, so that ceil () and the image
 * edge test give the same result.  They need IEEE float operations
 * without contraction into fma; the products use Dekker's splitting
 * instead of fma ().
 */
#pragma OPENCL FP_CONTRACT OFF

float2
df_two_sum (float a,
            float b)
{
  const float s  = a + b;
  const float bb = s - a;
  return (float2)(s, (a - (s - bb)) + (b - bb));
}

float2
df_split (float a)
{
  const float c  = 4097.0f * a;
  const float hi = c - (c - a);
  return (float2)(hi, a - hi);
}

float2
df_two_prod (float a,
             float b)
{
  const float  p  = a * b;
  const float2 as = df_split (a);
  const float2 bs = df_split (b);
  const float  e  = ((as.x * bs.x - p) + as.x * bs.y + as.y * bs.x) +
                    as.y * bs.y;
  return (float2)(p, e);
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
         __global const float4 *trig,
                                  const int     max_n,
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

  /* the host computed the table up to max_n, with a margin */
  n = min (n, max_n);
  trig += n * (n - 1) / 2 - 3;

  /* far above the rounding noise of the positions, see below */
  const float snap = (fabs ((float) x) + fabs ((float) y) +
                      fabs (xr2.x) + fabs (yr2.x)) * 0x1.0p-36f;

  float4 sum = (float4)0.0f;
  int count = 0;

  /* Iterate over the arc, rotating the offset of the pixel from the
   * center by theta in [angle / 2, -angle / 2), like the CPU code.
   * The middle sample (theta == 0) lands exactly on the pixel.
   * sin (theta) and cos (theta) - 1 come from a table computed in double
   * precision on the host, and the position is computed in double-float
   * precision before it is rounded to float like on the CPU, so samples
   * that lie exactly on the edge of the input (e.g. at 180 or 360
   * degrees around a center between pixels) are tested the same way.
   */
  for (int i = 0; i < n; i++)
    {
      const float4 t     = trig[i];
      const float2 s_val = t.xy;
      const float2 c_m1  = t.zw;

      const float2 px = df_add ((float2)((float) x, 0.0f),
                                df_add (df_mul (xr2, c_m1),
                                        -df_mul (yr2, s_val)));
      const float2 py = df_add ((float2)((float) y, 0.0f),
                                df_add (df_mul (xr2, s_val),
                                        df_mul (yr2, c_m1)));
      /* A sample exactly on the left or top edge of an input at 0 can
       * come out as a tiny positive or negative value, depending on the
       * rounding of sin () and cos (); float keeps it, so snap it to 0,
       * the exact result, like the CPU code does.  Elsewhere the
       * rounding to float hides that noise.
       */
      const float fx = fabs (px.x) < snap ? 0.0f : px.x;
      const float fy = fabs (py.x) < snap ? 0.0f : py.x;
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
