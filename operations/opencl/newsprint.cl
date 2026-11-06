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
 * the CPU code operation by operation: contractions are disabled, the
 * multiply-adds that the C compiler contracts (-ffp-contract=on) are
 * written as explicit fma (), divisions are corrected to be correctly
 * rounded, and cos/sin of the pattern angles are computed by the host
 * (so angle boost, which makes them vary per pixel, is not supported).
 */
#pragma OPENCL FP_CONTRACT OFF

#define MODEL_WHITE_ON_BLACK 0
#define MODEL_BLACK_ON_WHITE 1
#define MODEL_RGB            2
#define MODEL_CMYK           3

/* a / b, correctly rounded (OpenCL only guarantees 2.5 ulp) */
static float
div_rn (float a,
        float b)
{
  float q = a / b;
  float r = fma (-q, b, a);

  return fma (r, 1.0f / b, q);
}

/* hypot (a, b), correctly rounded in practice (OpenCL allows 4 ulp) */
static float
hypot_rn (float a,
          float b)
{
  float s  = a * a;
  float t  = b * b;
  float hi = s + t;
  float bb = hi - s;
  float lo = ((s - (hi - bb)) + (t - bb)) + fma (a, a, -s) + fma (b, b, -t);
  float r  = sqrt (hi);

  if (r == 0.0f)
    return 0.0f;

  return fma (fma (-r, r, hi) + lo, 0.5f / r, r);
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
  float width   = fma (period, 1.0f - turbulence,
                       (offset * period) * turbulence);
  float bwidth  = blocksize * width;
  float xi      = 0.5f;
  float yi      = 0.2f;
  int   count   = 0;
  int   in      = 0;

  x = fma (period, 2.0f, x);
  y = fma (period, 2.0f, y);

  for (int i = 0; i < max_aa_samples; i++)
    {
      float a, u, v, w, q, wphase, qphase;

      xi += 0x1.3c6ef4p-1f;
      xi -= floor (xi);
      yi += 0x1.888316p-2f;
      yi -= floor (yi);

      old_acc = acc;

      a = fma (width, -0.5f, x + xi);
      u = fma (-bwidth, floor (div_rn (a, bwidth)), a);
      a = fma (width, -0.5f, y + yi);
      v = fma (-bwidth, floor (div_rn (a, bwidth)), a);

      w = fma (vec0, u, vec1 * v);
      q = fma (vec1, u, vec0 * -v);

      w = fma (-width, floor (div_rn (w, width)), w);
      q = fma (-width, floor (div_rn (q, width)), q);

      wphase = fma (div_rn (w, width), 2.0f, -1.0f);
      qphase = fma (div_rn (q, width), 2.0f, -1.0f);

      if (pattern == 0) /* line */
        {
          if (fabs (wphase) < part_white)
            in++;
        }
      else if (pattern == 1) /* dot */
        {
          if (fma (qphase, qphase, wphase * wphase) <
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
  float bh = 1.0f - k;
  float bl = (1.0f - bh) - k;
  float q  = div_rn (a, bh);
  float r  = fma (-q, bh, a) - q * bl;

  return fma (r, 1.0f / bh, q);
}

/* (float) fma ((double) c, 1.0 - (double) k, (double) k) */
static float
mix_one_minus (float c,
               float k)
{
  float bh = 1.0f - k;
  float bl = (1.0f - bh) - k;
  float p  = c * bh;
  float pe = fma (c, bh, -p);
  float s  = p + k;
  float sb = s - p;
  float se = (p - (s - sb)) + (k - sb);

  return s + (se + pe + c * bl);
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

      k = fma (k, pullout_hi, k * pullout_lo);

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
