#pragma OPENCL FP_CONTRACT OFF

#define NEIGHBOURS 8
#define AXES       (NEIGHBOURS/2)

#define POW2(a) ((a)*(a))

#define GEN_METRIC(before, center, after) POW2((center) * (float4)(2.0f) - (before) - (after))

#define BAIL_CONDITION(new,original) (((new) > (original)))

#define SYMMETRY(a)  (NEIGHBOURS - (a) - 1)

#define O(u,v) (((u)+((v) * (src_stride))))

/* 1/count for count = 1..9, rounded by the compiler. */
__constant float reciprocal[10] = {
  0.0f, 1.0f, 1.0f / 2.0f, 1.0f / 3.0f, 1.0f / 4.0f,
  1.0f / 5.0f, 1.0f / 6.0f, 1.0f / 7.0f, 1.0f / 8.0f, 1.0f / 9.0f
};

/* Correctly rounded sum / count for count in 1..9.
 *
 * The result feeds hard comparisons in the next iteration, so it has to
 * match the CPU's IEEE division bit for bit; OpenCL 1.2 only guarantees
 * 2.5 ulp for '/' unless the device supports
 * -cl-fp32-correctly-rounded-divide-sqrt, which most mobile GPUs do not.
 *
 * q is within a few ulp of the quotient.  Splitting q into a 12-bit high
 * part and the remaining low bits makes both partial products with the
 * (4-bit) count exact, and the subtractions are exact as well, so r is the
 * exact remainder sum - q * count.  A quotient of floats is never exactly
 * halfway between two floats and here stays at least ulp/18 away from such
 * a midpoint, so the rounded q + r / count is the correctly rounded
 * quotient.
 *
 * Both parts of q are taken from its bits: ql is the difference of two
 * bit-masked floats with the same exponent, which is exact.  Written as
 * q - qh, a compiler that contracts across statements could turn it into
 * fma (sum, rcp, -qh) using the unrounded product, and r would then not be
 * the remainder for q.  For the same reason the result is formed as
 * qh + (ql + r * rcp) rather than q + r * rcp.  The only products that may
 * still be fused are exact (qh * cf, ql * cf) or so small (r * rcp) that
 * the rounding difference stays far below the ulp/18 margin, so the
 * result does not depend on FP contraction or fma.
 */
static float4
div_count (float4 sum,
           int4   count)
{
  float4 cf  = convert_float4 (count);
  float4 rcp = (float4) (reciprocal[count.x], reciprocal[count.y],
                         reciprocal[count.z], reciprocal[count.w]);
  float4 q   = sum * rcp;
  uint4  qb  = as_uint4 (q);
  float4 qh  = as_float4 (qb & (uint4) 0xfffff000u);
  float4 ql  = as_float4 (qb & (uint4) 0xff800fffu) -
               as_float4 (qb & (uint4) 0xff800000u);
  float4 r   = (sum - qh * cf) - ql * cf;
  float4 res = qh + (ql + r * rcp);

  /* inf / nan input: keep the plain product */
  return select (res, q, isnan (res));
}

__kernel void noise_reduction_cl (__global       float4 *src_buf,
                                  int src_stride,
                                  __global       float4 *dst_buf,
                                  int dst_stride)
{
    int gidx = get_global_id(0);
    int gidy = get_global_id(1);

    __global float4 *center_pix = src_buf + (gidy + 1) * src_stride + gidx + 1;
    int dst_offset = dst_stride * gidy + gidx;

    int offsets[NEIGHBOURS] = {
        O(-1, -1), O( 0, -1), O( 1, -1),
        O(-1,  0),            O( 1,  0),
        O(-1,  1), O( 0,  1), O( 1,  1)
    };

    float4 sum;
    int4   count;
    float4 cur;
    float4 metric_reference[AXES];

    for (int axis = 0; axis < AXES; axis++)
      {
        float4 before_pix = *(center_pix + offsets[axis]);
        float4 after_pix  = *(center_pix + offsets[SYMMETRY(axis)]);
        metric_reference[axis] = GEN_METRIC (before_pix, *center_pix, after_pix);
      }

    cur = sum = *center_pix;
    count = 1;

    for (int direction = 0; direction < NEIGHBOURS; direction++)
      {
        float4 pix   = *(center_pix + offsets[direction]);
        float4 value = (pix + cur) * (0.5f);
        int    axis;
        int4   mask = {1, 1, 1, 0};

        for (axis = 0; axis < AXES; axis++)
          {
            float4 before_pix = *(center_pix + offsets[axis]);
            float4 after_pix  = *(center_pix + offsets[SYMMETRY(axis)]);

            float4 metric_new = GEN_METRIC (before_pix,
                                            value,
                                            after_pix);
            mask = select (mask, (int4)0, BAIL_CONDITION (metric_new, metric_reference[axis]));
          }

        sum   += select ((float4)0.0f,  value, mask > 0);
        count += select ((int4)0,     (int4)1, mask > 0);
      }
    dst_buf[dst_offset]   = div_count (sum, count);
    dst_buf[dst_offset].w = cur.w;
}
__kernel void transfer(__global float4 * in,
              int               in_width,
              __global float4 * out)
{
    int gidx = get_global_id(0);
    int gidy = get_global_id(1);
    int width = get_global_size(0);
    out[gidy * width + gidx] = in[gidy * in_width + gidx];
}
