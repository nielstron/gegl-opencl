__kernel void
gegl_sepia (__global const float4 *in,
            __global       float4 *out,
            float4                 red,
            float4                 green,
            float4                 blue)
{
  const size_t i = get_global_id (0);
  const float4 p = in[i];

  out[i] = (float4) (dot (p.xyz, red.xyz),
                    dot (p.xyz, green.xyz),
                    dot (p.xyz, blue.xyz), p.w);
}
