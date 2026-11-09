__kernel void
gegl_saturation (__global const float4 *in,
                 __global       float4 *out,
                 float4                 luminance,
                 float                  scale,
                 float                  rscale)
{
  const size_t i = get_global_id (0);
  const float4 p = in[i];
  const float gray = dot (p.xyz, luminance.xyz) * rscale;

  out[i] = (float4) (gray + p.xyz * scale, p.w);
}
