/* A small error in the circle distance can change the lens blur disc's
 * integer span. Keep the residual of x*x + y*y to approach the CPU's
 * double-precision hypot without requiring FP64 on the device.
 */
static inline float
focus_blur_hypot (float x,
                  float y)
{
  float major = fmax (fabs (x), fabs (y));
  float minor = fmin (fabs (x), fabs (y));
  float a = major * major;
  float b = minor * minor;
  float sum = a + b;

  if (sum >= FLT_MIN && isfinite (sum))
    {
      float error = ((a - sum) + b) +
                    (fma (major, major, -a) + fma (minor, minor, -b));
      float approx = sqrt (sum);

      return approx + (fma (-approx, approx, sum) + error) / (2.0f * approx);
    }

  return hypot (x, y);
}

static inline float
vignette_strength (int   x,
                   int   y,
                   int   midx,
                   int   midy,
                   int   shape,
                   float scale,
                   float cost,
                   float sint,
                   float gamma,
                   float length,
                   float radius0,
                   float rdiff,
                   bool  accurate)
{
  float strength = 0.0f;
  float u,v,costy,sinty;
  sinty = sint * (y-midy) - midx;
  costy = cost * (y-midy) + midy;

  u = cost * (x-midx) - sinty;
  v = sint * (x-midx) + costy;

  if (length == 0.0f)
    strength = 0.0f;
  else
    {
      switch (shape)
        {
          case 0:
          strength = accurate ? focus_blur_hypot ((u-midx) / scale, v-midy)
                              : hypot ((u-midx) / scale, v-midy);
          break;

          case 1:
          strength = fmax (fabs(u-midx)/scale, fabs(v-midy));
          break;

          case 2:
          strength = fabs (u-midx) / scale + fabs(v-midy);
          break;

          case 3:
          strength = fabs(v-midy);
          break;

          case 4:
          strength = fabs (u-midx) / scale;
          break;
        }
      strength /= length;
      strength = (strength-radius0) / rdiff;
    }

  if (strength < 0.0f) strength = 0.0f;
  if (strength > 1.0f) strength = 1.0f;

  if (gamma > 1.9999f && gamma < 2.0001f)
    strength *= strength;
  else if (gamma != 1.0f)
    strength = pow(strength, gamma);

  return strength;
}

__kernel void vignette_cl (__global const float4 *in,
                           __global       float4 *out,
                                          float4 color,
                                          float  scale,
                                          float  cost,
                                          float  sint,
                                          int    roi_x,
                                          int    roi_y,
                                          int    midx,
                                          int    midy,
                                          int    o_shape,
                                          float  gamma,
                                          float  length,
                                          float  radius0,
                                          float  rdiff)
{
  int gidx = get_global_id (0);
  int gidy = get_global_id (1);
  size_t gid = gidx + gidy * get_global_size (0);
  float strength = vignette_strength (gidx + roi_x, gidy + roi_y,
                                      midx, midy, o_shape, scale, cost,
                                      sint, gamma, length, radius0, rdiff, false);

  out[gid] = in[gid]*(1.0f-strength) + color * strength;
}

__kernel void focus_blur_mask (__global float *mask,
                               int             roi_x,
                               int             roi_y,
                               int             midx,
                               int             midy,
                               int             shape,
                               float           scale,
                               float           cost,
                               float           sint,
                               float           gamma,
                               float           length,
                               float           radius0,
                               float           rdiff,
                               int             valid_x,
                               int             valid_y,
                               int             valid_right,
                               int             valid_bottom,
                               float4          luminance)
{
  int gidx = get_global_id (0);
  int gidy = get_global_id (1);
  size_t gid = gidx + gidy * get_global_size (0);
  int x = gidx + roi_x;
  int y = gidy + roi_y;

  if (x < valid_x || x >= valid_right || y < valid_y || y >= valid_bottom)
    mask[gid] = 0.0f;
  else
    {
      float strength = vignette_strength (x, y, midx, midy, shape, scale, cost,
                                          sint, gamma, length, radius0, rdiff, true);

      /* The graph converts the opaque grayscale vignette from linear sRGB
       * to Y float. Match colors.cl's Babl luminance conversion. */
      mask[gid] = fma (strength, luminance.z,
                       fma (strength, luminance.x, strength * luminance.y));
    }
}
