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
inline float4
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

inline float4
unpremultiply (float4 v)
{
  const float alpha = v.w;
  const float used  = (alpha <= ALPHA_FLOOR && alpha >= -ALPHA_FLOOR) ?
                      ALPHA_FLOOR : alpha;
  const float recip = 1.0f / used;

  return (float4) (v.xyz * recip, alpha);
}

#define CHUNK_SIZE 512

inline float4
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
                 float                  centre_x,
                 float                  centre_y,
                 float                  mult_sq,
                 float                  mult_qd,
                 float                  rescale,
                 float                  brighten_amount,
                 float                  norm,
                 int                    unpremultiply_output)
{
  const int   gidx  = get_global_id (0);
  const int   gidy  = get_global_id (1);
  const int   xx    = roi_x + gidx;
  const int   yy    = roi_y + gidy;
  const int4  area  = areas[((yy - result_y) / CHUNK_SIZE) * chunks_x +
                            (xx - result_x) / CHUNK_SIZE];
  float       off_x, off_y, radius_sq, radius_mult, mag, sx, sy;
  float       brighten, dx, dy;
  float       um1, u, up1, up2, vm1, v, vp1, vp2;
  float4      verts[4];
  float4      result;
  int         x_int, y_int, i;

  /* lens_get_source_coord () */
  off_x       = xx - centre_x;
  off_y       = yy - centre_y;
  radius_sq   = (off_x * off_x + off_y * off_y) * norm;
  radius_mult = radius_sq * mult_sq + radius_sq * radius_sq * mult_qd;
  mag         = radius_mult;
  radius_mult = rescale * (1.0f + radius_mult);
  sx          = centre_x + radius_mult * off_x;
  sy          = centre_y + radius_mult * off_y;

  brighten = pow (max (1.0f + mag * brighten_amount, 0.0f), 2.4f);

  x_int = (int) floor (sx);
  dx    = sx - x_int;
  y_int = (int) floor (sy);
  dy    = sy - y_int;

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
