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

/* The kernels work on linear RGBA float and do the CIE Lab conversion
 * inline, following babl's CIE model: the RGB to XYZ matrices of the
 * space are adapted to D50 and passed in by the host.
 */

#define D50_WHITE_REF_X 0.964202880f
#define D50_WHITE_REF_Z 0.824905400f
#define LAB_EPSILON     (216.0f / 24389.0f)
#define LAB_KAPPA       (24389.0f / 27.0f)

/* sRGB transfer function, see babl/base/util.h */
static float
linear_to_gamma (float value)
{
  if (value > 0.003130804954f)
    return 1.055f * powr (value, 1.0f / 2.4f) - 0.055f;
  return 12.92f * value;
}

static float
lab_f (float t)
{
  return t > LAB_EPSILON ? cbrt (t) : (LAB_KAPPA * t + 16.0f) / 116.0f;
}

static float
lab_f_inv (float f)
{
  float f3 = f * f * f;

  return f3 > LAB_EPSILON ? f3 : (116.0f * f - 16.0f) / LAB_KAPPA;
}

static float3
rgb_to_lab (float4 rgb,
            float4 m0,
            float4 m1,
            float4 m2)
{
  float x  = dot (rgb.xyz, m0.xyz) / D50_WHITE_REF_X;
  float y  = dot (rgb.xyz, m1.xyz);
  float z  = dot (rgb.xyz, m2.xyz) / D50_WHITE_REF_Z;
  float fx = lab_f (x);
  float fy = lab_f (y);
  float fz = lab_f (z);

  return (float3) (116.0f * fy - 16.0f,
                   500.0f * (fx - fy),
                   200.0f * (fy - fz));
}

/* Each work-group writes (min chroma, -max chroma) of its pixels to
 * out[group_id]; the host combines the per-group results.
 */
__kernel void
chroma_min_max (__global const float4 *in,
                       __global       float2 *out,
                       __local        float2 *aux,
                                      int     n_pixels,
                                      float4  m0,
                                      float4  m1,
                                      float4  m2)
{
  int    gid   = get_global_id (0);
  int    gsize = get_global_size (0);
  int    lid   = get_local_id (0);
  int    lsize = get_local_size (0);
  float2 acc   = (float2) (FLT_MAX);
  int    it;

  for (it = gid; it < n_pixels; it += gsize)
    {
      float3 lab    = rgb_to_lab (in[it], m0, m1, m2);
      float  chroma = sqrt (lab.y * lab.y + lab.z * lab.z);

      acc = fmin (acc, (float2) (chroma, -chroma));
    }

  aux[lid] = acc;
  barrier (CLK_LOCAL_MEM_FENCE);

  for (it = lsize / 2; it > 0; it >>= 1)
    {
      if (lid < it)
        aux[lid] = fmin (aux[lid], aux[lid + it]);
      barrier (CLK_LOCAL_MEM_FENCE);
    }

  if (lid == 0)
    out[get_group_id (0)] = aux[0];
}

__kernel void
color_enhance (__global const float4 *in,
               __global       float4 *out,
                              float   min_chroma,
                              float   delta,
                              float4  m0,
                              float4  m1,
                              float4  m2,
                              float4  i0,
                              float4  i1,
                              float4  i2,
                              int     to_gamma)
{
  int    gid    = get_global_id (0);
  float4 in_v   = in[gid];
  float3 lab    = rgb_to_lab (in_v, m0, m1, m2);
  float  chroma = sqrt (lab.y * lab.y + lab.z * lab.z);
  float  hue    = atan2 (lab.z, lab.y);
  float  fx, fy, fz;
  float3 xyz, rgb;

  chroma = (chroma - min_chroma) / delta * 100.0f;

  lab.y = chroma * cos (hue);
  lab.z = chroma * sin (hue);

  fy = (lab.x + 16.0f) / 116.0f;
  fx = fy + lab.y / 500.0f;
  fz = fy - lab.z / 200.0f;

  xyz.x = lab_f_inv (fx) * D50_WHITE_REF_X;
  xyz.y = lab.x > LAB_KAPPA * LAB_EPSILON ? fy * fy * fy : lab.x / LAB_KAPPA;
  xyz.z = lab_f_inv (fz) * D50_WHITE_REF_Z;

  rgb = (float3) (dot (xyz, i0.xyz),
                  dot (xyz, i1.xyz),
                  dot (xyz, i2.xyz));

  if (to_gamma)
    rgb = (float3) (linear_to_gamma (rgb.x),
                    linear_to_gamma (rgb.y),
                    linear_to_gamma (rgb.z));

  out[gid] = (float4) (rgb, in_v.w);
}
