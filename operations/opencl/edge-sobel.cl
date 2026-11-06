/* The CPU implementation reads the source rectangle (result grown by one
   pixel on each side) into a buffer and takes the output pixel (x, y) as
   centred on source pixel (x, y), i.e. on image pixel (x - 1, y - 1).
   Neighbours to the left of / above that buffer are clamped into it.

   To give the same result for every chunk the iterator hands us, the input
   is requested with a halo of 2 pixels on the left and top and none on the
   right and bottom, and min_x / min_y say whether this chunk touches the
   left / top edge of the processed rectangle, where the CPU code clamps.  */

static inline float4 get_pixel (global const float4 *in,
                                int                  x,
                                int                  y,
                                int                  min_x,
                                int                  min_y,
                                int                  width)
{
    x = max (x, min_x);
    y = max (y, min_y);
    return in[x + y * width];
}

kernel void kernel_edgesobel (global const float4 *in,
                              global       float4 *out,
                              const int horizontal,
                              const int vertical,
                              const int keep_sign,
                              const int has_alpha,
                              const int min_x,
                              const int min_y)
{
    const int gidx = get_global_id (0);
    const int gidy = get_global_id (1);

    float4 hor_grad = 0.0f;
    float4 ver_grad = 0.0f;
    float4 gradient = 0.0f;

    const int dst_width = get_global_size (0);
    const int src_width = dst_width + 2;

    const int x = gidx + 1;
    const int y = gidy + 1;

    const float4 pix_fl = get_pixel (in, x - 1, y - 1, min_x, min_y, src_width);
    const float4 pix_fm = get_pixel (in, x,     y - 1, min_x, min_y, src_width);
    const float4 pix_fr = get_pixel (in, x + 1, y - 1, min_x, min_y, src_width);
    const float4 pix_ml = get_pixel (in, x - 1, y,     min_x, min_y, src_width);
    const float4 pix_mm = in[x + y * src_width];
    const float4 pix_mr = get_pixel (in, x + 1, y,     min_x, min_y, src_width);
    const float4 pix_bl = get_pixel (in, x - 1, y + 1, min_x, min_y, src_width);
    const float4 pix_bm = get_pixel (in, x,     y + 1, min_x, min_y, src_width);
    const float4 pix_br = get_pixel (in, x + 1, y + 1, min_x, min_y, src_width);
    if (horizontal)
      {
        hor_grad +=
            (-1.0f * pix_fl + 1.0f * pix_fr) +
            (-2.0f * pix_ml + 2.0f * pix_mr) +
            (-1.0f * pix_bl + 1.0f * pix_br);
      }
    if (vertical)
      {
        ver_grad +=
            ( 1.0f * pix_fl) + ( 2.0f * pix_fm) + ( 1.0f * pix_fr) +
            (-1.0f * pix_bl) + (-2.0f * pix_bm) + (-1.0f * pix_br);
      }

    if (horizontal && vertical)
      {
        /* sqrt(32.0) = 5.656854 */
        gradient = sqrt (hor_grad * hor_grad + ver_grad * ver_grad) / 5.656854f;
      }
    else if (keep_sign)
      {
        gradient = 0.5f + (hor_grad + ver_grad) / 8.0f;
      }
    else
      {
        gradient = fabs (hor_grad + ver_grad) / 4.0f;
      }

    if (has_alpha)
      gradient.w = pix_mm.w;
    else
      gradient.w = 1.0f;

    out[gidx + gidy * dst_width] = gradient;
}
