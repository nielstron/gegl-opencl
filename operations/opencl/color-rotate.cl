#pragma OPENCL FP_CONTRACT OFF

/* Split 2*pi so hue angles round as they do on the CPU, including at slice
 * boundaries where choosing the wrong branch changes the output sharply. */
#define TWO_PI_HI 0x1.921fb6p+2f
#define TWO_PI_LO -0x1.777a5dp-23f

/* OpenCL's float division can be off by several ulps. Near a discontinuity,
 * divide the binary32 significands and round the quotient like the CPU. */
static float
divide (float a, float b, bool precise)
{
  const float approx = a / b;
  if (!precise || !isfinite (approx) || fabs (approx) < FLT_MIN)
    return approx;

  int a_exp, b_exp;
  const uint a_significand = (uint) ldexp (frexp (fabs (a), &a_exp), 24);
  const uint b_significand = (uint) ldexp (frexp (fabs (b), &b_exp), 24);
  const int normalize = a_significand < b_significand;
  const int exponent = a_exp - b_exp - normalize;

  if (exponent < -126)
    return approx;

  const ulong numerator = (ulong) a_significand << (23 + normalize);
  uint quotient = (uint) (numerator / b_significand);
  const uint remainder = (uint) (numerator - (ulong) quotient * b_significand);

  if (2 * remainder > b_significand ||
      (2 * remainder == b_significand && (quotient & 1)))
    quotient++;

  return copysign (ldexp ((float) quotient, exponent - 23), approx);
}

static float3
rgb_to_hsv (float3 rgb, bool precise)
{
  const float v = max (max (rgb.x, rgb.y), rgb.z);
  const float delta = v - min (min (rgb.x, rgb.y), rgb.z);
  float h = 0.0f;
  float s = 0.0f;

  if (delta != 0.0f)
    {
      s = divide (delta, v, precise);

      if (rgb.x == v)
        {
          h = divide (rgb.y - rgb.z, delta, precise);
          if (h < 0.0f)
            h += 6.0f;
        }
      else if (rgb.y == v)
        h = 2.0f + divide (rgb.z - rgb.x, delta, precise);
      else
        h = 4.0f + divide (rgb.x - rgb.y, delta, precise);

      h = divide (h, 6.0f, precise);
    }

  return (float3) (h, s, v);
}

static float3
hsv_to_rgb (float3 hsv)
{
  const float s = hsv.y;
  const float v = hsv.z;
  float h = hsv.x * 6.0f;

  if (s == 0.0f)
    return (float3) (v);

  if (h >= 6.0f)
    h -= 6.0f;

  const int sector = (int) h;
  const float frac = h - sector;
  const float w = v * (1.0f - s);
  const float q = v * (1.0f - s * frac);
  const float t = v * (1.0f - s * (1.0f - frac));

  switch (sector)
    {
    case 0:  return (float3) (v, t, w);
    case 1:  return (float3) (q, v, w);
    case 2:  return (float3) (w, v, t);
    case 3:  return (float3) (w, q, v);
    case 4:  return (float3) (t, w, v);
    default: return (float3) (v, w, q);
    }
}

__kernel void
gegl_color_rotate (__global const float4 *in,
                   __global       float4 *out,
                   float4                 angles,
                   float2                 wrapped_range,
                   float4                 gray,
                   int                    gray_mode)
{
  const size_t i = get_global_id (0);
  const float4 pixel = in[i];
  float3 hsv = rgb_to_hsv (pixel.xyz, false);
  float hue = fma (hsv.x, TWO_PI_HI, hsv.x * TWO_PI_LO);
  const float lo = min (angles.x, angles.y);
  const float hi = max (angles.x, angles.y);
  const float slope = (angles.w - angles.z) / (angles.y - angles.x);
  bool rotate = true;

  /* Guard both ends of the ordinary and wrapped hue range. For a very
   * narrow range, the slope can also amplify errors inside the slice. */
  if ((hsv.y != 0.0f && fabs (hsv.y - gray.x) <= 8.0f * FLT_EPSILON * fabs (hsv.y)) ||
      (hsv.x != 0.0f &&
       (fabs (hue - lo) <= 8e-6f || fabs (hue - hi) <= 8e-6f ||
        fabs (hue - wrapped_range.x) <= 8e-6f || fabs (hue - wrapped_range.y) <= 8e-6f ||
        (fabs (slope) > 1024.0f &&
         ((lo <= hue && hue <= hi) || (wrapped_range.x <= hue && hue <= wrapped_range.y))))))
    {
      hsv = rgb_to_hsv (pixel.xyz, true);
      hue = fma (hsv.x, TWO_PI_HI, hsv.x * TWO_PI_LO);
    }

  if (hsv.y <= gray.x)
    {
      rotate = gray_mode == 1;
      if (gray_mode != 0)
        {
          hsv.xy = gray.yz;
          hue = fma (hsv.x, TWO_PI_HI, hsv.x * TWO_PI_LO);
        }
    }

  if (rotate)
    {
      if (lo <= hue && hue <= hi)
        hue = angles.z + slope * (hue - angles.x);
      else if (wrapped_range.x <= hue && hue <= wrapped_range.y)
        hue = angles.z + slope * ((hue + (TWO_PI_HI - angles.x)) + TWO_PI_LO);

      if (hue < 0.0f)
        hue += TWO_PI_HI;
      else if (hue >= TWO_PI_HI)
        hue -= TWO_PI_HI;

      hsv.x = hue / TWO_PI_HI;
    }

  out[i] = (float4) (hsv_to_rgb (hsv), pixel.w);
}
