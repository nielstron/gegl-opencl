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

/* Port of gegl:lens-distortion.  The CPU code processes its result in
 * chunks of CHUNK_SIZE x CHUNK_SIZE pixels and reads the source pixels of
 * a chunk from a precomputed area; source pixels outside that area (but
 * inside the input) are read with a linear GeglSampler at integer
 * coordinates.  'areas' holds the area of each chunk so that both cases
 * come out the same here.
 */

/* The source coordinates are computed with pairs of floats (hi + lo,
 * "float-float" arithmetic, about 44 bits of precision) instead of
 * doubles.  Single floats are not precise enough: the fractional part of
 * a source coordinate in the thousands would be off by up to ~3e-4
 * pixels, which "brighten" (a gain of several hundred in the corners)
 * makes visible.  These algorithms need every operation to be rounded
 * on its own, hence no contraction into fma.
 */
#pragma OPENCL FP_CONTRACT OFF

float2
ff_two_sum (float a,
            float b)
{
  float s  = a + b;
  float bb = s - a;
  float t1 = s - bb;
  float t2 = a - t1;
  float t3 = b - bb;

  return (float2) (s, t2 + t3);
}

float2
ff_quick_two_sum (float a,
                  float b)
{
  float s = a + b;
  float t = s - a;

  return (float2) (s, b - t);
}

float2
ff_split (float a)
{
  float t  = 4097.0f * a;
  float t1 = t - a;
  float hi = t - t1;

  return (float2) (hi, a - hi);
}

float2
ff_two_prod (float a,
             float b)
{
  float  p  = a * b;
  float2 as = ff_split (a);
  float2 bs = ff_split (b);
  float  e1 = as.x * bs.x;
  float  e2 = as.x * bs.y;
  float  e3 = as.y * bs.x;
  float  e4 = as.y * bs.y;
  float  e;

  e = e1 - p;
  e = e + e2;
  e = e + e3;
  e = e + e4;

  return (float2) (p, e);
}

float2
ff_add (float2 a,
        float2 b)
{
  float2 s = ff_two_sum (a.x, b.x);
  float  l = a.y + b.y;

  return ff_quick_two_sum (s.x, s.y + l);
}

float2
ff_mul (float2 a,
        float2 b)
{
  float2 p  = ff_two_prod (a.x, b.x);
  float  c1 = a.x * b.y;
  float  c2 = a.y * b.x;
  float  c  = c1 + c2;

  return ff_quick_two_sum (p.x, p.y + c);
}

/* lens_get_source_coord () for pixel (xx, yy): returns the integer part
 * of the source position in 'pos', its fractional part in 'frac' and
 * the magnitude used for "brighten".
 */
float
lens_source_coord (int     xx,
                   int     yy,
                   float2  centre_x,
                   float2  centre_y,
                   float2  mult_sq,
                   float2  mult_qd,
                   float2  rescale,
                   float2  norm,
                   int2   *pos,
                   float2 *frac)
{
  float2 off_x, off_y, radius_sq, radius_mult, sx, sy;
  float  fx, fy, mag;

  off_x       = ff_add ((float2) ((float) xx, 0.0f), -centre_x);
  off_y       = ff_add ((float2) ((float) yy, 0.0f), -centre_y);
  radius_sq   = ff_add (ff_mul (off_x, off_x), ff_mul (off_y, off_y));
  radius_sq   = ff_mul (radius_sq, norm);
  radius_mult = ff_add (ff_mul (radius_sq, mult_sq),
                        ff_mul (ff_mul (radius_sq, radius_sq), mult_qd));
  mag         = radius_mult.x + radius_mult.y;
  radius_mult = ff_mul (rescale, ff_add (radius_mult, (float2) (1.0f, 0.0f)));
  sx          = ff_add (centre_x, ff_mul (radius_mult, off_x));
  sy          = ff_add (centre_y, ff_mul (radius_mult, off_y));

  /* floor () of hi + lo; keep the integer part in range of an int */
  sx.x = clamp (sx.x, -1.0e9f, 1.0e9f);
  sy.x = clamp (sy.x, -1.0e9f, 1.0e9f);
  fx   = floor (sx.x);
  fy   = floor (sy.x);
  if (fx == sx.x && sx.y < 0.0f)
    fx -= 1.0f;
  if (fy == sy.x && sy.y < 0.0f)
    fy -= 1.0f;

  *pos  = (int2) ((int) fx, (int) fy);
  *frac = (float2) ((sx.x - fx) + sx.y, (sy.x - fy) + sy.y);

  return mag;
}

/* Abyss policies, matching GeglAbyssPolicy */
#define ABYSS_NONE  0
#define ABYSS_CLAMP 1
#define ABYSS_LOOP  2
#define ABYSS_BLACK 3
#define ABYSS_WHITE 4

/* Fetch one pixel of the input like GeglSampler does: coordinates
 * outside the buffer's abyss rectangle are handled according to the
 * abyss policy, everything else is read from the uploaded source
 * rectangle 'srect' (x, y, width, height).
 */
float4
fetch (__global const float4 *src,
       int4                   srect,
       int4                   abyss,
       int                    policy,
       int                    x,
       int                    y)
{
  if (x < abyss.x || y < abyss.y ||
      x >= abyss.x + abyss.z || y >= abyss.y + abyss.w)
    {
      switch (policy)
        {
        case ABYSS_CLAMP:
          x = clamp (x, abyss.x, abyss.x + abyss.z - 1);
          y = clamp (y, abyss.y, abyss.y + abyss.w - 1);
          break;
        case ABYSS_LOOP:
          x = abyss.x + (((x - abyss.x) % abyss.z) + abyss.z) % abyss.z;
          y = abyss.y + (((y - abyss.y) % abyss.w) + abyss.w) % abyss.w;
          break;
        case ABYSS_BLACK:
          return (float4) (0.0f, 0.0f, 0.0f, 1.0f);
        case ABYSS_WHITE:
          return (float4) (1.0f);
        default:
          return (float4) (0.0f);
        }
    }

  x = clamp (x - srect.x, 0, srect.z - 1);
  y = clamp (y - srect.y, 0, srect.w - 1);

  return src[y * srect.z + x];
}

#define FETCH(x, y) fetch (src, srect, abyss, policy, (x), (y))

/* "RaGaBaA float" to "RGBA float", like babl does it */
#define ALPHA_FLOOR (1.0f / 65536.0f)

float4
unpremultiply (float4 v)
{
  const float alpha = v.w;
  const float used  = (alpha <= ALPHA_FLOOR && alpha >= -ALPHA_FLOOR) ?
                      ALPHA_FLOOR : alpha;
  const float recip = 1.0f / used;

  return (float4) (v.xyz * recip, alpha);
}

#define CHUNK_SIZE 512

float4
lens_get_pixel (__global const float4 *src,
                int4                   srect,
                int4                   abyss,
                int4                   boundary,
                int4                   area,
                float4                 background,
                int                    x,
                int                    y)
{
  const int policy = ABYSS_CLAMP;

  if (x < boundary.x || x >= boundary.x + boundary.z ||
      y < boundary.y || y >= boundary.y + boundary.w)
    return background;

  if (x >= area.x && x < area.x + area.z &&
      y >= area.y && y < area.y + area.w)
    return FETCH (x, y);

  /* gegl_sampler_get () with GEGL_SAMPLER_LINEAR at (x, y), which
   * averages four pixels and returns them as "RGBA float" */
  return unpremultiply (0.25f * FETCH (x,     y    ) +
                        0.25f * FETCH (x - 1, y    ) +
                        0.25f * FETCH (x,     y - 1) +
                        0.25f * FETCH (x - 1, y - 1));
}

__kernel void
lens_distortion (__global const float4 *src,
                 int4                   srect,
                 int4                   abyss,
                 __global       float4 *dst,
                 int                    roi_x,
                 int                    roi_y,
                 int                    result_x,
                 int                    result_y,
                 __global const int4   *areas,
                 int                    chunks_x,
                 int4                   boundary,
                 float4                 background,
                 float2                 centre_x,
                 float2                 centre_y,
                 float2                 mult_sq,
                 float2                 mult_qd,
                 float2                 rescale,
                 float2                 norm,
                 float                  brighten_amount,
                 int                    unpremultiply_output)
{
  const int   gidx  = get_global_id (0);
  const int   gidy  = get_global_id (1);
  const int   xx    = roi_x + gidx;
  const int   yy    = roi_y + gidy;
  const int4  area  = areas[((yy - result_y) / CHUNK_SIZE) * chunks_x +
                            (xx - result_x) / CHUNK_SIZE];
  float       mag, brighten, dx, dy;
  int2        pos;
  float2      frac;
  float       um1, u, up1, up2, vm1, v, vp1, vp2;
  float4      verts[4];
  float4      result;
  int         x_int, y_int, i;

  mag = lens_source_coord (xx, yy, centre_x, centre_y, mult_sq, mult_qd,
                           rescale, norm, &pos, &frac);

  brighten = pow (max (1.0f + mag * brighten_amount, 0.0f), 2.4f);

  x_int = pos.x;
  dx    = frac.x;
  y_int = pos.y;
  dy    = frac.y;

  /* lens_cubic_interpolate () */
  um1 = ((-0.5f * dx + 1.0f) * dx - 0.5f) * dx;
  u   = (1.5f * dx - 2.5f) * dx * dx + 1.0f;
  up1 = ((-1.5f * dx + 2.0f) * dx + 0.5f) * dx;
  up2 = (0.5f * dx - 0.5f) * dx * dx;

  vm1 = ((-0.5f * dy + 1.0f) * dy - 0.5f) * dy;
  v   = (1.5f * dy - 2.5f) * dy * dy + 1.0f;
  vp1 = ((-1.5f * dy + 2.0f) * dy + 0.5f) * dy;
  vp2 = (0.5f * dy - 0.5f) * dy * dy;

  for (i = 0; i < 4; i++)
    {
      const int x = x_int - 1 + i;

      verts[i] =
        vm1 * lens_get_pixel (src, srect, abyss, boundary, area, background,
                              x, y_int - 1) +
        v   * lens_get_pixel (src, srect, abyss, boundary, area, background,
                              x, y_int    ) +
        vp1 * lens_get_pixel (src, srect, abyss, boundary, area, background,
                              x, y_int + 1) +
        vp2 * lens_get_pixel (src, srect, abyss, boundary, area, background,
                              x, y_int + 2);
    }

  result = um1 * verts[0] + u * verts[1] + up1 * verts[2] + up2 * verts[3];
  result.xyz *= brighten;
  result = clamp (result, 0.0f, 1.0f);

  if (unpremultiply_output)
    result = unpremultiply (result);

  dst[gidy * get_global_size (0) + gidx] = result;
}


/* The range of source pixels the samples of a part of a row of the
 * result touch, so that only that part of the input has to be uploaded.
 * Each work item handles up to 'seg' pixels of row 'roi_y + gid (1)'
 * and stores (min x, min y, max x, max y) of the integer parts of their
 * source positions, limited to 'limit' (x0, y0, x1, y1).
 */
__kernel void
lens_distortion_bounds (__global int4 *bounds,
                        int            roi_x,
                        int            roi_y,
                        int            roi_width,
                        int            seg,
                        int4           limit,
                        float2         centre_x,
                        float2         centre_y,
                        float2         mult_sq,
                        float2         mult_qd,
                        float2         rescale,
                        float2         norm)
{
  const int gidx = get_global_id (0);
  const int gidy = get_global_id (1);
  const int x0   = roi_x + gidx * seg;
  const int x1   = min (x0 + seg, roi_x + roi_width);
  const int yy   = roi_y + gidy;
  int4      b    = (int4) (limit.z, limit.w, limit.x, limit.y);
  int2      pos;
  float2    frac;
  int       xx;

  for (xx = x0; xx < x1; xx++)
    {
      lens_source_coord (xx, yy, centre_x, centre_y, mult_sq, mult_qd,
                         rescale, norm, &pos, &frac);

      pos = clamp (pos, limit.xy, limit.zw);
      b   = (int4) (min (b.xy, pos), max (b.zw, pos));
    }

  bounds[gidy * get_global_size (0) + gidx] = b;
}
