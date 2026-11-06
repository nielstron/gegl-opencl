/* This file is an image processing operation for GEGL
 *
 * GEGL is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * GEGL is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GEGL; if not, see <https://www.gnu.org/licenses/>.
 */

/* Halftoning thresholds every anti-aliasing sample, so a one ulp
 * difference in the sample phase can flip a sample and change the output
 * by up to a quarter of the range. The arithmetic below therefore mirrors
 * the CPU code (which is compiled without contraction) operation by
 * operation, with every operation separately rounded: divisions and
 * hypot are corrected to be correctly rounded, gegl_fmodf () with its
 * truncating floor is reproduced exactly, including for zero and
 * huge quotients, and cos/sin of the pattern angles are computed by the
 * host (so angle boost, which makes them vary per pixel, is not
 * supported). The error-free products use fma () only where the device
 * says it is fast, and Dekker's product otherwise.
 */
#pragma OPENCL FP_CONTRACT OFF

#define MODEL_WHITE_ON_BLACK 0
#define MODEL_BLACK_ON_WHITE 1
#define MODEL_RGB            2
#define MODEL_CMYK           3

/* (p, e) with p = a * b rounded and p + e == a * b exactly */
static float2
two_prod (float a,
          float b)
{
  float p = a * b;
#ifdef FP_FAST_FMAF
  return (float2) (p, fma (a, b, -p));
#else
  float ca = 4097.0f * a;
  float cb = 4097.0f * b;
  float ah = ca - (ca - a);
  float bh = cb - (cb - b);
  float al = a - ah;
  float bl = b - bh;

  return (float2) (p, ((ah * bh - p) + ah * bl + al * bh) + al * bl);
#endif
}

/* a / b, correctly rounded (OpenCL only guarantees 2.5 ulp) */
static float
div_rn (float a,
        float b)
{
  float  q = a / b;
  float2 p;

  if (b == 0.0f)
    return (a == 0.0f || isnan (a)) ? NAN
                                    : copysign (INFINITY, a) *
                                      copysign (1.0f, b);

  if (q == 0.0f || ! isfinite (q))
    return q;

  p = two_prod (q, b);

  return q + ((a - p.x) - p.y) * (1.0f / b);
}

/* gegl_floorf (): truncation through (int), which saturates (and gives
 * 0 for NaN) on ARM, then a wrapping int subtraction
 */
static float
gegl_floor (float x)
{
  int i = convert_int_sat (x);

  return (float) as_int (as_uint (i) - (((float) i > x) ? 1u : 0u));
}

/* gegl_fmodf (): x - y * gegl_floorf (x / y) */
static float
gegl_fmod (float x,
           float y)
{
  return x - y * gegl_floor (div_rn (x, y));
}

/* hypot (a, b), correctly rounded in practice like the CPU's */
static float
hypot_rn (float a,
          float b)
{
  float2 s, t;
  float2 rr;
  float  hi, bb, lo, r;

  if (isinf (a) || isinf (b))
    return INFINITY;
  if (isnan (a) || isnan (b))
    return NAN;

  s  = two_prod (a, a);
  t  = two_prod (b, b);
  hi = s.x + t.x;
  bb = hi - s.x;
  lo = ((s.x - (hi - bb)) + (t.x - bb)) + (s.y + t.y);
  r  = sqrt (hi);

  if (r == 0.0f)
    return 0.0f;

  rr = two_prod (r, r);

  return r + (((hi - rr.x) - rr.y) + lo) * (0.5f / r);
}

static float
spachrotyze (float x,
             float y,
             float part_white,
             float offset,
             int   pattern,
             float period,
             float turbulence,
             float blocksize,
             int   max_aa_samples,
             float vec0,
             float vec1)
{
  float acc     = 0.0f;
  float old_acc = 0.0f;
  float width   = period * (1.0f - turbulence) +
                  (period * offset) * turbulence;
  float bwidth  = blocksize * width;
  float xi      = 0.5f;
  float yi      = 0.2f;
  int   count   = 0;
  int   in      = 0;

  x = x + period * 2.0f;
  y = y + period * 2.0f;

  for (int i = 0; i < max_aa_samples; i++)
    {
      float u, v, w, q, wphase, qphase;

      xi += 0x1.3c6ef4p-1f;
      xi -= floor (xi);
      yi += 0x1.888316p-2f;
      yi -= floor (yi);

      old_acc = acc;

      u = gegl_fmod ((x + xi) - 0.5f * width, bwidth);
      v = gegl_fmod ((y + yi) - 0.5f * width, bwidth);

      w = vec0 * u + vec1 * v;
      q = vec1 * u - vec0 * v;

      w = gegl_fmod (w, width);
      q = gegl_fmod (q, width);

      wphase = div_rn (w, width) * 2.0f - 1.0f;
      qphase = div_rn (q, width) * 2.0f - 1.0f;

      if (pattern == 0) /* line */
        {
          if (fabs (wphase) < part_white)
            in++;
        }
      else if (pattern == 1) /* dot */
        {
          if (qphase * qphase + wphase * wphase <
              (part_white * part_white) * 2.0f)
            in++;
        }
      else if (pattern == 2) /* diamond */
        {
          if ((fabs (wphase) + fabs (qphase)) * 0.5f < part_white)
            in++;
        }
      else if (pattern == 3) /* dot-to-diamond-to-dot */
        {
          float ax = fabs (wphase);
          float ay = fabs (qphase);
          float d;

          if (ax + ay > 1.0f)
            d = 2.0f - hypot_rn (1.0f - ax, 1.0f - ay);
          else
            d = hypot_rn (ax, ay);

          if (d * 0.5f < part_white)
            in++;
        }
      else if (pattern == 4) /* cross */
        {
          float part_white2 = part_white * part_white;

          if (fabs (wphase) < part_white2 || fabs (qphase) < part_white2)
            in++;
        }

      count++;

      acc = div_rn ((float) in, (float) count);

      if (i > 3 && fabs (acc - old_acc) < 0.23f)
        break;
    }

  return acc;
}

/* (float) ((double) a / (1.0 - (double) k)) */
static float
div_one_minus (float a,
               float k)
{
  float  bh = 1.0f - k;
  float  bl = (1.0f - bh) - k;
  float  q  = div_rn (a, bh);
  float2 p;

  if (q == 0.0f || ! isfinite (q))
    return q;

  p = two_prod (q, bh);

  return q + (((a - p.x) - p.y) - q * bl) * (1.0f / bh);
}

/* (float) ((double) k * pullout), pullout == hi + lo */
static float
mul_pullout (float k,
             float hi,
             float lo)
{
  float2 p = two_prod (k, hi);

  return p.x + (p.y + k * lo);
}

/* (float) ((double) c * (1.0 - (double) k) + (double) k) */
static float
mix_one_minus (float c,
               float k)
{
  float  bh = 1.0f - k;
  float  bl = (1.0f - bh) - k;
  float2 p  = two_prod (c, bh);
  float  s  = p.x + k;
  float  sb = s - p.x;
  float  se = (p.x - (s - sb)) + (k - sb);

  return s + (se + p.y + c * bl);
}

static float
linear_to_gamma (float value)
{
  if (value > 0.003130804954f)
    return 1.055f * powr (value, 1.0f / 2.4f) - 0.055f;
  return 12.92f * value;
}

/* Patterns, periods and the cos/sin of the pattern angles are
 * indexed by ink: black (or gray), red/cyan, green/magenta, blue/yellow.
 */
__kernel void
newsprint (__global const float4 *in,
           __global       float4 *out,
                          int     color_model,
                          int4    patterns,
                          float4  periods,
                          float4  vec0s,
                          float4  vec1s,
                          float   turbulence,
                          float   blocksize,
                          int     aa_samples,
                          float   pullout_hi,
                          float   pullout_lo,
                          int     roi_x,
                          int     roi_y,
                          int     roi_width,
                          int     to_gamma)
{
  int    gid    = get_global_id (0);
  int    iy     = gid / roi_width;
  float  x      = (float) (roi_x + gid - iy * roi_width);
  float  y      = (float) (roi_y + iy);
  float4 in_v   = in[gid];
  float  chroma = fabs (in_v.x - in_v.y);
  float4 out_v;

#define SPACHROTYZE(part_white, ink)                                     \
  spachrotyze (x, y, part_white, chroma,                                 \
               patterns.s##ink, periods.s##ink, turbulence, blocksize,   \
               aa_samples, vec0s.s##ink, vec1s.s##ink)

  if (color_model == MODEL_WHITE_ON_BLACK)
    {
      out_v.xyz = (float3) (SPACHROTYZE (in_v.y, 0));
    }
  else if (color_model == MODEL_BLACK_ON_WHITE)
    {
      out_v.xyz = (float3) (1.0f - SPACHROTYZE (1.0f - in_v.y, 0));
    }
  else if (color_model == MODEL_RGB)
    {
      out_v.x = SPACHROTYZE (in_v.x, 1);
      out_v.y = SPACHROTYZE (in_v.y, 2);
      out_v.z = SPACHROTYZE (in_v.z, 3);
    }
  else
    {
      float c = 1.0f - in_v.x;
      float m = 1.0f - in_v.y;
      float j = 1.0f - in_v.z;
      float k = 1.0f;

      /* the CPU code compares against the row, not yellow */
      if (c < k) k = c;
      if (m < k) k = m;
      if (y < k) k = y;

      k = mul_pullout (k, pullout_hi, pullout_lo);

      if (k < 1.0f)
        {
          c = div_one_minus (c - k, k);
          m = div_one_minus (m - k, k);
          j = div_one_minus (j - k, k);
        }
      else
        {
          c = m = j = 1.0f;
        }

      c = SPACHROTYZE (c, 1);
      m = SPACHROTYZE (m, 2);
      j = SPACHROTYZE (j, 3);
      k = SPACHROTYZE (k, 0);

      if (k < 1.0f)
        {
          c = mix_one_minus (c, k);
          m = mix_one_minus (m, k);
          j = mix_one_minus (j, k);
        }
      else
        {
          c = m = j = 1.0f;
        }

      out_v.xyz = (float3) (1.0f - c, 1.0f - m, 1.0f - j);
    }

#undef SPACHROTYZE

  if (to_gamma)
    out_v.xyz = (float3) (linear_to_gamma (out_v.x),
                          linear_to_gamma (out_v.y),
                          linear_to_gamma (out_v.z));

  out_v.w  = in_v.w;
  out[gid] = out_v;
}
