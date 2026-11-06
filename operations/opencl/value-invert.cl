/* Shortcut of RGB -> HSV, v = 1.0 - v, HSV -> RGB; see value-invert.c */
static float4
value_invert (float4 in_v)
{
  float value = fmax (in_v.x, fmax (in_v.y, in_v.z));
  float minv  = fmin (in_v.x, fmin (in_v.y, in_v.z));
  float delta = value - minv;
  float4 out_v;

  if (value == 0.0f || delta == 0.0f)
    {
      out_v.xyz = (float3) (1.0f - value);
    }
  else
    {
      out_v.xyz = (1.0f - value) * in_v.xyz / value;
    }

  out_v.w = in_v.w;

  return out_v;
}

__kernel void gegl_value_invert (__global const float4 *in,
                                 __global       float4 *out)
{
  int gid = get_global_id (0);

  out[gid] = value_invert (in[gid]);
}

/* R'G'B'A u8 in, R'G'B'A float out, which saves the conversion pass and
 * the float buffer on the input side.  The conversion gives the same result
 * as babl: k / 255.0f correctly rounded (refined with an fma, as the
 * division need not be).
 */
__kernel void gegl_value_invert_u8 (__global const uchar4 *in,
                                    __global       float4 *out)
{
  int    gid = get_global_id (0);
  float4 k   = convert_float4 (in[gid]);
  float4 v   = k * (1.0f / 255.0f);

  v        = fma (fma (-v, 255.0f, k), (float4) (1.0f / 255.0f), v);
  out[gid] = value_invert (v);
}
