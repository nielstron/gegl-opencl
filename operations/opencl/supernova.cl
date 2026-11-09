__kernel void
gegl_supernova (__global const float4 *in,
                 __global       float4 *out,
                 __constant     float4 *spokes,
                 float2                 origin,
                 int                    width,
                 float                  radius,
                 int                    count)
{
  const size_t i = get_global_id (0);
  const float4 pixel = in[i];
  const float2 xy = (origin + (float2) (i % width, i / width)) / radius;
  const float l = length (xy);
  const float t = (0.5f * atan2pi (xy.x, xy.y) + 0.51f) * count;
  const int spoke = (int) floor (t);
  const float4 color = mix (spokes[spoke % count],
                            spokes[(spoke + 1) % count], t - spoke);
  const float strength = 0.9f / (l + 0.001f);
  const float nova_alpha = clamp (strength, 0.0f, 1.0f);
  const float alpha = pixel.w + (1.0f - pixel.w) * nova_alpha;
  const float ratio = alpha != 0.0f ? nova_alpha / alpha : 0.0f;
  const float3 rgb = strength > 1.0f ?
                     clamp (color.xyz * strength, 0.0f, 1.0f) :
                     pixel.xyz * (1.0f - ratio) + color.xyz * ratio;
  const float light = clamp (color.w * color.w * strength, 0.0f, 1.0f);

  out[i] = (float4) (clamp (rgb + light, 0.0f, 1.0f), alpha);
}
