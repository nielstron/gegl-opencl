__kernel void
gegl_invert_gamma_rgba_u8 (__global const uchar4 *in,
                           __global       uchar4 *out)
{
  const size_t i = get_global_id (0);

  out[i] = in[i] ^ (uchar4) (255, 255, 255, 0);
}

__kernel void
gegl_invert_gamma_rgb_u8 (__global const uchar *in,
                          __global       uchar *out)
{
  const size_t i = get_global_id (0);

  vstore3 (~vload3 (i, in), i, out);
}
