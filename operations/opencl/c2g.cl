#define ANGLE_PRIME 95273
#define RADIUS_PRIME 29537

/* (a * b) % p for a, b < p < 2^17, using only 32-bit unsigned arithmetic
 * (no 64-bit integers, no overflow). */
static uint mulmod_prime(uint a, uint b, uint p)
{
    uint r = (a * (b >> 8)) % p;
    return (r * 256u + a * (b & 255u)) % p;
}

/* The CPU draws the sample sequence from a global counter; here each pixel
 * gets its own start offset into the lookup tables and then consumes the
 * sequence in order over all its iterations, as one CPU thread would.  The
 * offset is pixel * iterations * samples reduced modulo the table size
 * without ever forming the (possibly > 2^31) product.
 */
static int table_start(uint pix, uint per_pixel, uint p)
{
    return (int)mulmod_prime(pix % p, per_pixel % p, p);
}

static void sample_min_max(const __global   float4 *src_buf,
                                            int     src_width,
                                            int     src_height,
                           const __global   float  *radiuses,
                           const __global   float  *lut_cos,
                           const __global   float  *lut_sin,
                                            int     x,
                                            int     y,
                                            int4    valid,
                                            int2    origin,
                                            int     radius,
                                            int     samples,
                                            float4 *min,
                                            float4 *max,
                                            int    *angle_state,
                                            int    *radius_state)
{
    float4 best_min;
    float4 best_max;
    float4 center_pix = *(src_buf + src_width * y + x);
    int i;

    best_min = center_pix;
    best_max = center_pix;

    int  angle_no  = *angle_state;
    int  radius_no = *radius_state;

    /* Absolute image coordinates of this pixel, the CPU samples in those. */
    float xa = (float)(x + origin.x);
    float ya = (float)(y + origin.y);

    for(i=0; i<samples; i++)
    {
        /* Like the CPU: a sample outside the image is redrawn (bounded here
         * by one full cycle of the radius table, which contains radii small
         * enough to land inside), a transparent sample is redrawn at most
         * 'samples' times. */
        int transparent_retries = samples;
        int draws;

        for (draws = 0; draws < RADIUS_PRIME; draws++)
        {
            int angle = angle_no++;
            float rmag = radiuses[radius_no++] * radius;

            if (angle_no  >= ANGLE_PRIME)
                angle_no   = 0;
            if (radius_no >= RADIUS_PRIME)
                radius_no  = 0;

            /* Truncate in absolute coordinates, as the CPU does. */
            int u = (int)(xa + rmag * lut_cos[angle]);
            int v = (int)(ya + rmag * lut_sin[angle]);

            if (u < valid.x || u >= valid.z || v < valid.y || v >= valid.w)
                continue;

            float4 pixel = *(src_buf + (src_width * (v - origin.y) +
                                        (u - origin.x)));
            if (pixel.w > 0.0f)
            {
                best_min = pixel < best_min ? pixel : best_min;
                best_max = pixel > best_max ? pixel : best_max;
                break;
            }
            if (--transparent_retries <= 0)
                break;
        }
    }

    *angle_state  = angle_no;
    *radius_state = radius_no;
    (*min).xyz = best_min.xyz;
    (*max).xyz = best_max.xyz;
}

static void compute_envelopes(const __global  float4 *src_buf,
                                       int     src_width,
                                       int     src_height,
                       const __global  float  *radiuses,
                       const __global  float  *lut_cos,
                       const __global  float  *lut_sin,
                                       int     x,
                                       int     y,
                                       int4    valid,
                                       int2    origin,
                                       int     radius,
                                       int     samples,
                                       int     iterations,
                                       float4 *min_envelope,
                                       float4 *max_envelope)
{
    float4 range_sum = 0;
    float4 relative_brightness_sum = 0;
    float4 pixel = *(src_buf + src_width * y + x);

    uint pix       = (uint)(src_width * y + x);
    uint per_pixel = (uint)iterations * (uint)samples;
    int  angle_no  = table_start(pix, per_pixel, ANGLE_PRIME);
    int  radius_no = table_start(pix, per_pixel, RADIUS_PRIME);

    int i;
    for(i =0; i<iterations; i++)
    {
        float4 min,max;
        float4 range, relative_brightness;

        sample_min_max(src_buf, src_width, src_height,
                        radiuses, lut_cos, lut_sin, x, y,
                        valid, origin, radius, samples, &min, &max,
                       &angle_no, &radius_no);
        range = max - min;
        relative_brightness = range <= 0.0f ?
                               0.5f : (pixel - min) / range;
        relative_brightness_sum += relative_brightness;
        range_sum += range;
    }

    float4 relative_brightness = relative_brightness_sum / (float4)(iterations);
    float4 range = range_sum / (float4)(iterations);

    if(max_envelope)
        *max_envelope = pixel + (1.0f - relative_brightness) * range;

    if(min_envelope)
        *min_envelope = pixel - relative_brightness * range;
}

__kernel void c2g(const __global float4 *src_buf,
                                 int     src_width,
                                 int     src_height,
                  const __global float  *radiuses,
                  const __global float  *lut_cos,
                  const __global float  *lut_sin,
                        __global float2 *dst_buf,
                                 int     radius,
                                 int     samples,
                                 int     iterations,
                                 int     enhance_shadows,
                                 int     origin_x,
                                 int     origin_y,
                                 int     image_width,
                                 int     image_height)
{
    int gidx = get_global_id(0);
    int gidy = get_global_id(1);

    int x = gidx + radius;
    int y = gidy + radius;

    int src_offset = (src_width * y + x);
    int dst_offset = gidx + get_global_size(0) * gidy;
    float4 min_env, max_env;

    /* Sampleable area in absolute coordinates: the CPU's image area
     * [0,image_width) x [0,image_height), clipped to the input chunk. */
    int2 origin = (int2)(origin_x, origin_y);
    int4 valid  = (int4)(max(0, origin_x), max(0, origin_y),
                         min(image_width,  origin_x + src_width),
                         min(image_height, origin_y + src_height));

    compute_envelopes(src_buf, src_width, src_height,
                      radiuses, lut_cos, lut_sin, x, y,
                      valid, origin, radius, samples, iterations, &min_env, &max_env);

    float4 pixel = *(src_buf + src_offset);

    float nominator=0, denominator=0;
    /* Match the CPU default: without shadow enhancement the black point
     * is zero, rather than the sampled minimum envelope. */
    float4 relative = enhance_shadows ? pixel - min_env : pixel;
    float4 t1 = relative * relative;
    float4 t2 = (pixel - max_env) * (pixel - max_env);

    nominator   = t1.x + t1.y + t1.z;
    denominator = t2.x + t2.y + t2.z;

    nominator   = sqrt(nominator);
    denominator = sqrt(denominator);
    denominator = nominator + denominator;

    dst_buf[dst_offset].x = (denominator > 0.000f)
                             ? (nominator / denominator) : 0.5f;
    dst_buf[dst_offset].y =  src_buf[src_offset].w;
}
