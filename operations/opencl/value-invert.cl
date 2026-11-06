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

/* k / 255.0f correctly rounded, as babl converts u8 to float (refined
 * with an fma, as the division need not be)
 */
static float4
u8_to_float (uchar4 in_v)
{
  float4 k = convert_float4 (in_v);
  float4 v = k * (1.0f / 255.0f);

  return fma (fma (-v, 255.0f, k), (float4) (1.0f / 255.0f), v);
}

/* R'G'B'A u8 in, R'G'B'A float out, which saves the conversion pass and
 * the float buffer on the input side.
 */
__kernel void gegl_value_invert_u8 (__global const uchar4 *in,
                                    __global       float4 *out)
{
  int gid = get_global_id (0);

  out[gid] = value_invert (u8_to_float (in[gid]));
}

/* R'G'B'A u8 in and out, for an output buffer that is u8 itself; rounds
 * to u8 like babl does.
 */
__kernel void gegl_value_invert_u8_u8 (__global const uchar4 *in,
                                       __global       uchar4 *out)
{
  int    gid   = get_global_id (0);
  uchar4 in_v  = in[gid];
  float4 out_v = value_invert (u8_to_float (in_v));
  uchar4 res;

  res      = convert_uchar4_sat (clamp (out_v, 0.0f, 1.0f) * 255.0f + 0.5f);
  res.w    = in_v.w;
  out[gid] = res;
}
