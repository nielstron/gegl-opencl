typedef struct
{
  int2   center;
  float  radius;
  int    type;
  float4 color;
} Effect;

__kernel void
gegl_lens_flare (__global const float4 *in,
                 __global       float4 *out,
                 __constant     Effect *effects,
                 int2                   origin,
                 int                    width)
{
  const size_t i = get_global_id (0);
  const int2 xy = origin + (int2) (i % width, i / width);
  float4 pixel = in[i];
  float hyp = length (convert_float2 (xy - effects[0].center));

  for (int j = 0; j < 24; j++)
    {
      const Effect effect = effects[j];
      const float size = effect.radius;
      const int type = effect.type;
      float amount;

      if (j >= 5)
        {
          const float2 delta = convert_float2 (xy - effect.center);
          const float limit = size * (type == 4 ? 1.04f : 1.0f);

          if (any (fabs (delta) > limit))
            continue;
          hyp = length (delta);
        }

      switch (type)
        {
        case 0:
          amount = (size - hyp) / size;
          break;
        case 1:
          amount = (size - hyp) / size;
          if (amount > 0.0f)
            amount *= amount;
          break;
        case 2:
          amount = min ((size - hyp) / (size * 0.15f), 1.0f);
          break;
        case 3:
          amount = (size - hyp) / (size * 0.12f);
          if (amount > 1.0f)
            amount = 1.0f - amount * 0.12f;
          break;
        default:
          amount = 1.0f - fabs ((hyp - size) / (size * (type == 4 ? 0.04f : 0.07f)));
          break;
        }

      if (amount > 0.0f)
        pixel.xyz += (1.0f - pixel.xyz) * amount * effect.color.xyz;
    }

  out[i] = pixel;
}
