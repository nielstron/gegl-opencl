/* The CPU implementation reads the source rectangle (result grown by one
   pixel on each side) into a buffer and takes the output pixel (x, y) as
   centred on source pixel (x, y), i.e. on image pixel (x - 1, y - 1).
   Neighbours to the left of / above that buffer are clamped into it.

   To give the same result for every chunk the iterator hands us, the input
   is requested with a halo of 2 pixels on the left and top and none on the
   right and bottom, and min_x / min_y say whether this chunk touches the
   left / top edge of the processed rectangle, where the CPU code clamps.

   The input and the output are either "RGBA float", or "R'G'B'A u8",
   which is converted here instead of in separate passes of the buffer
   iterator.  lut holds tables computed with babl on the host, so that the
   conversions match the CPU path:

     lut[  0..255]  "R'G'B'A u8" color value -> "RGBA float"
     lut[256..511]  "R'G'B'A u8" alpha value -> "RGBA float"
     lut[512..767]  "RGBA float" color -> "R'G'B'A u8": lut[512 + k] is
                    the smallest float that is converted to k or more
     lut[768..1023] the same for alpha

   Every work-group first converts the input pixels it needs, its own
   block grown by one pixel on each side, into local memory, so that each
   input pixel is converted about once instead of up to nine times.  */

#define LUT_EXPAND_COLOR   0
#define LUT_EXPAND_ALPHA   256
#define LUT_QUANT_COLOR    512
#define LUT_QUANT_ALPHA    768

static inline uchar quantize (float              value,
                              local const float *lut)
{
  int i = 0;

  /* comparisons with NaN are false, giving 0 */
  i += (lut[i + 128] <= value) ? 128 : 0;
  i += (lut[i +  64] <= value) ?  64 : 0;
  i += (lut[i +  32] <= value) ?  32 : 0;
  i += (lut[i +  16] <= value) ?  16 : 0;
  i += (lut[i +   8] <= value) ?   8 : 0;
  i += (lut[i +   4] <= value) ?   4 : 0;
  i += (lut[i +   2] <= value) ?   2 : 0;
  i += (lut[i +   1] <= value) ?   1 : 0;

  return (uchar) i;
}

kernel void kernel_edgesobel (global const void  *in,
                              global       void  *out,
                              global const float *lut,
                              local        float4 *tile,
                              local        float  *quant_lut,
                              const int           dst_width,
                              const int           dst_height,
                              const int           horizontal,
                              const int           vertical,
                              const int           keep_sign,
                              const int           has_alpha,
                              const int           min_x,
                              const int           min_y,
                              const int           in_u8,
                              const int           out_u8)
{
    const int lx      = get_local_id (0);
    const int ly      = get_local_id (1);
    const int lw      = get_local_size (0);
    const int lh      = get_local_size (1);
    const int tile_w  = lw + 2;
    const int tile_h  = lh + 2;
    const int src_w   = dst_width + 2;
    const int src_h   = dst_height + 2;
    const int x0      = get_group_id (0) * lw;
    const int y0      = get_group_id (1) * lh;
    const int gidx    = x0 + lx;
    const int gidy    = y0 + ly;
    const int lid     = lx + ly * lw;
    const int n_local = lw * lh;

    float4 hor_grad = 0.0f;
    float4 ver_grad = 0.0f;
    float4 gradient;
    int    i;

    /* tile pixel (tx, ty) is source pixel (x0 + tx, y0 + ty), clamped */
    for (i = lid; i < tile_w * tile_h; i += n_local)
      {
        const int tx    = i % tile_w;
        const int ty    = i / tile_w;
        const int sx    = clamp (x0 + tx, min_x, src_w - 1);
        const int sy    = clamp (y0 + ty, min_y, src_h - 1);
        const int index = sx + sy * src_w;

        if (in_u8)
          {
            const uchar4 p = ((global const uchar4 *) in)[index];

            tile[i] = (float4) (lut[LUT_EXPAND_COLOR + p.x],
                                lut[LUT_EXPAND_COLOR + p.y],
                                lut[LUT_EXPAND_COLOR + p.z],
                                lut[LUT_EXPAND_ALPHA + p.w]);
          }
        else
          {
            tile[i] = ((global const float4 *) in)[index];
          }
      }

    if (out_u8)
      for (i = lid; i < 512; i += n_local)
        quant_lut[i] = lut[LUT_QUANT_COLOR + i];

    barrier (CLK_LOCAL_MEM_FENCE);

    if (gidx >= dst_width || gidy >= dst_height)
      return;

    {
      /* the output pixel is centred on tile pixel (lx + 1, ly + 1) */
      local const float4 *t = tile + lx + ly * tile_w;

      const float4 pix_fl = t[0];
      const float4 pix_fm = t[1];
      const float4 pix_fr = t[2];
      const float4 pix_ml = t[tile_w];
      const float4 pix_mm = t[tile_w + 1];
      const float4 pix_mr = t[tile_w + 2];
      const float4 pix_bl = t[2 * tile_w];
      const float4 pix_bm = t[2 * tile_w + 1];
      const float4 pix_br = t[2 * tile_w + 2];

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

      if (out_u8)
        {
          const int src_index = gidx + 1 + (gidy + 1) * src_w;
          uchar4    result;

          result.x = quantize (gradient.x, quant_lut);
          result.y = quantize (gradient.y, quant_lut);
          result.z = quantize (gradient.z, quant_lut);

          /* an 8-bit alpha converts to float and back unchanged */
          if (! has_alpha)
            result.w = 255;
          else if (in_u8)
            result.w = ((global const uchar4 *) in)[src_index].w;
          else
            result.w = quantize (pix_mm.w, quant_lut + 256);

          ((global uchar4 *) out)[gidx + gidy * dst_width] = result;
        }
      else
        {
          gradient.w = has_alpha ? pix_mm.w : 1.0f;

          ((global float4 *) out)[gidx + gidy * dst_width] = gradient;
        }
    }
}
