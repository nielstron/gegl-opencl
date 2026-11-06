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

/* The kernels work on linear RGBA float and do the HSV conversion inline,
 * following babl's HSV model, which is defined on the linear RGB of the
 * space encoded with the sRGB TRC (regardless of the space's own TRC).
 * Like babl, the linear segment of the TRC also applies to negative values,
 * pixels whose value is below 1e-10 (including negative ones) get zero
 * saturation and hue, and a channel within 1e-10 of the maximum selects
 * the hue sector.
 */

#define HSV_EPSILON 1e-10f

static float3
linear_to_srgb (float3 v)
{
  return (float3) (
    v.x > 0.003130804954f ? 1.055f * pow (v.x, 1.0f / 2.4f) - 0.055f
                          : 12.92f * v.x,
    v.y > 0.003130804954f ? 1.055f * pow (v.y, 1.0f / 2.4f) - 0.055f
                          : 12.92f * v.y,
    v.z > 0.003130804954f ? 1.055f * pow (v.z, 1.0f / 2.4f) - 0.055f
                          : 12.92f * v.z);
}

static float3
srgb_to_linear (float3 v)
{
  return (float3) (
    v.x > 0.04045f ? pow ((v.x + 0.055f) / 1.055f, 2.4f) : v.x / 12.92f,
    v.y > 0.04045f ? pow ((v.y + 0.055f) / 1.055f, 2.4f) : v.y / 12.92f,
    v.z > 0.04045f ? pow ((v.z + 0.055f) / 1.055f, 2.4f) : v.z / 12.92f);
}

static float2
rgb_to_sv (float4 lin)
{
  float3 in     = linear_to_srgb (lin.xyz);
  float  max_v  = fmax (in.x, fmax (in.y, in.z));
  float  min_v  = fmin (in.x, fmin (in.y, in.z));
  float  chroma = max_v - min_v;
  float  s      = (max_v < HSV_EPSILON) ? 0.0f : chroma / max_v;

  return (float2) (s, max_v);
}

/* Each work-group writes (min s, min v, -max s, -max v) of its pixels to
 * out[group_id]; the host combines the per-group results.
 */
__kernel void
hsv_sv_min_max (__global const float4 *in,
                __global       float4 *out,
                __local        float4 *aux,
                               int     n_pixels)
{
  int    gid   = get_global_id (0);
  int    gsize = get_global_size (0);
  int    lid   = get_local_id (0);
  int    lsize = get_local_size (0);
  float4 acc   = (float4) (FLT_MAX);
  int    it;

  for (it = gid; it < n_pixels; it += gsize)
    {
      float2 sv = rgb_to_sv (in[it]);

      acc = fmin (acc, (float4) (sv, -sv));
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
stretch_contrast_hsv (__global const float4 *in,
                      __global       float4 *out,
                                     float   slo,
                                     float   sdiff,
                                     float   vlo,
                                     float   vdiff)
{
  int    gid    = get_global_id (0);
  float4 in_v   = (float4) (linear_to_srgb (in[gid].xyz), in[gid].w);
  float  max_v  = fmax (in_v.x, fmax (in_v.y, in_v.z));
  float  min_v  = fmin (in_v.x, fmin (in_v.y, in_v.z));
  float  chroma = max_v - min_v;
  float  hue    = 0.0f;
  float  sat    = (max_v < HSV_EPSILON) ? 0.0f : chroma / max_v;
  float  value  = max_v;
  float  h_tmp, x, m;
  float4 out_v;

  if (sat >= HSV_EPSILON)
    {
      if (fabs (in_v.x - max_v) < HSV_EPSILON)
        {
          hue = (in_v.y - in_v.z) / chroma;
          if (hue < 0.0f)
            hue += 6.0f;
        }
      else if (fabs (in_v.y - max_v) < HSV_EPSILON)
        hue = 2.0f + (in_v.z - in_v.x) / chroma;
      else
        hue = 4.0f + (in_v.x - in_v.y) / chroma;

      hue /= 6.0f;
    }

  sat   = (sat   - slo) / sdiff;
  value = (value - vlo) / vdiff;

  chroma = value * sat;
  h_tmp  = hue * 6.0f;
  x      = chroma * (1.0f - fabs (fmod (h_tmp, 2.0f) - 1.0f));
  m      = value - chroma;

  if (h_tmp < 1.0f)
    out_v = (float4) (chroma, x, 0.0f, 0.0f);
  else if (h_tmp < 2.0f)
    out_v = (float4) (x, chroma, 0.0f, 0.0f);
  else if (h_tmp < 3.0f)
    out_v = (float4) (0.0f, chroma, x, 0.0f);
  else if (h_tmp < 4.0f)
    out_v = (float4) (0.0f, x, chroma, 0.0f);
  else if (h_tmp < 5.0f)
    out_v = (float4) (x, 0.0f, chroma, 0.0f);
  else
    out_v = (float4) (chroma, 0.0f, x, 0.0f);

  out_v.xyz = srgb_to_linear (out_v.xyz + m);
  out_v.w   = in_v.w;

  out[gid] = out_v;
}
