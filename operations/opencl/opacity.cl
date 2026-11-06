__kernel void gegl_opacity_RaGaBaA_float (__global const float4     *in,
                                          __global const float      *aux,
                                          __global       float4     *out,
                                          float value)
{
  int gid = get_global_id(0);
  float4 in_v  = in [gid];
  float  aux_v = (aux)? aux[gid] : 1.0f;
  float4 out_v;
  out_v = in_v * aux_v * value;
  out[gid]  =  out_v;
}
__kernel void gegl_opacity_RGBA_float (__global const float4     *in,
                                       __global const float      *aux,
                                       __global       float4     *out,
                                       float value)
{
  int gid = get_global_id(0);
  float4 in_v  = in [gid];
  float  aux_v = (aux)? aux[gid] : 1.0f;
  float4 out_v = (float4)(in_v.x, in_v.y, in_v.z, in_v.w * aux_v * value);
  out[gid]  =  out_v;
}
__kernel void gegl_opacity_YaA_float (__global const float2     *in,
                                      __global const float      *aux,
                                      __global       float2     *out,
                                      float value)
{
  int gid = get_global_id(0);
  float2 in_v  = in [gid];
  float  aux_v = (aux)? aux[gid] : 1.0f;
  float2 out_v;
  out_v = in_v * aux_v * value;
  out[gid]  =  out_v;
}
__kernel void gegl_opacity_YA_float (__global const float2     *in,
                                     __global const float      *aux,
                                     __global       float2     *out,
                                     float value)
{
  int gid = get_global_id(0);
  float2 in_v  = in [gid];
  float  aux_v = (aux)? aux[gid] : 1.0f;
  float2 out_v = (float2)(in_v.x, in_v.y * aux_v * value);
  out[gid]  =  out_v;
}
