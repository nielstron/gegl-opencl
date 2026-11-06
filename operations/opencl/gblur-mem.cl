/* This file is an image processing operation for GEGL
 *
 * GEGL is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * GEGL is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GEGL; if not, see <https://www.gnu.org/licenses/>.
 */

/* FIR gaussian blur of a dense width x height image along its rows
 * (horizontal) or columns, following fir_blur_1D () in gblur-1d.c, for
 * 1, 2 and 4 float components per pixel. Pixels outside the image are
 * the nearest edge pixel (clamp_edges) or zero.
 *
 * The IIR kernels are the ones of gblur-1d.cl, which is built into the
 * same program.
 */
#define GBLUR_MEM_FIR(NAME, TYPE)                                       \
__kernel void NAME (const global TYPE  *src,                            \
                          global TYPE  *dst,                            \
                    const global float *cmatrix,                        \
                    const        int    clen,                           \
                    const        int    width,                          \
                    const        int    height,                         \
                    const        int    horizontal,                     \
                    const        int    clamp_edges)                    \
{                                                                       \
  const int x = get_global_id (0);                                      \
  const int y = get_global_id (1);                                      \
                                                                        \
  if (x >= width || y >= height)                                        \
    return;                                                             \
                                                                        \
  const int len   = horizontal ? width : height;                        \
  const int step  = horizontal ? 1 : width;                             \
  const int line  = horizontal ? y * width : x;                         \
  const int start = (horizontal ? x : y) - clen / 2;                    \
  TYPE      acc   = 0.0f;                                               \
  int       m;                                                          \
                                                                        \
  for (m = 0; m < clen; m++)                                            \
    {                                                                   \
      int p = start + m;                                                \
                                                                        \
      if (p < 0 || p >= len)                                            \
        {                                                               \
          if (! clamp_edges)                                            \
            continue;                                                   \
                                                                        \
          p = clamp (p, 0, len - 1);                                    \
        }                                                               \
                                                                        \
      acc += src[line + p * step] * cmatrix[m];                         \
    }                                                                   \
                                                                        \
  dst[y * width + x] = acc;                                             \
}

GBLUR_MEM_FIR (gblur_mem_fir_1, float)
GBLUR_MEM_FIR (gblur_mem_fir_2, float2)
GBLUR_MEM_FIR (gblur_mem_fir_4, float4)

/* Copies a chunk_width wide chunk of pixels to (x0, y0) of an image that
 * is stride pixels wide, optionally premultiplying the color components
 * with the last one, the alpha, as babl does.
 */
#define GBLUR_MEM_ALPHA_FLOOR (1.0f / 65536.0f)

#define GBLUR_MEM_USED_ALPHA(a)                                         \
  (((a) > GBLUR_MEM_ALPHA_FLOOR || (a) < -GBLUR_MEM_ALPHA_FLOOR) ?      \
   (a) : GBLUR_MEM_ALPHA_FLOOR)

__kernel void gblur_mem_put_1 (const global float *chunk,
                                     global float *image,
                               const        int    chunk_width,
                               const        int    x0,
                               const        int    y0,
                               const        int    stride,
                               const        int    premultiply)
{
  const int x = get_global_id (0);
  const int y = get_global_id (1);

  image[(y0 + y) * stride + x0 + x] = chunk[y * chunk_width + x];
}

__kernel void gblur_mem_put_2 (const global float2 *chunk,
                                     global float2 *image,
                               const        int     chunk_width,
                               const        int     x0,
                               const        int     y0,
                               const        int     stride,
                               const        int     premultiply)
{
  const int x = get_global_id (0);
  const int y = get_global_id (1);
  float2    v = chunk[y * chunk_width + x];

  if (premultiply)
    v.x *= GBLUR_MEM_USED_ALPHA (v.y);

  image[(y0 + y) * stride + x0 + x] = v;
}

__kernel void gblur_mem_put_4 (const global float4 *chunk,
                                     global float4 *image,
                               const        int     chunk_width,
                               const        int     x0,
                               const        int     y0,
                               const        int     stride,
                               const        int     premultiply)
{
  const int x = get_global_id (0);
  const int y = get_global_id (1);
  float4    v = chunk[y * chunk_width + x];

  if (premultiply)
    v.xyz *= GBLUR_MEM_USED_ALPHA (v.w);

  image[(y0 + y) * stride + x0 + x] = v;
}

/* The IIR passes of gblur-1d.cl filter every line in a single work-item,
 * which leaves most of a GPU idle and makes the time grow with the
 * length of the lines. These kernels split every line into n_segs
 * segments of seg_len pixels that are filtered in parallel, with the
 * same recursion as iir_young_causal () and iir_young_anticausal ().
 *
 * The first segment of the causal pass and the last one of the
 * anti-causal pass start from the boundary values as those kernels do.
 * Every other segment starts its recursion warmup pixels before the
 * segment, from the steady state of the pixel there; warmup is chosen so
 * that the effect of the start values decays below float precision
 * before the segment begins. A segment whose warm-up would reach the
 * start of the line starts from the boundary values instead.
 *
 * The anti-causal pass reads the causal result outside its segment, so
 * it must not run in place.
 */
__kernel void gblur_mem_iir_causal (const global float *src_buf,
                                          global float *dst_buf,
                                          global float *state_buf,
                                    const        int     len,
                                    const        int     n_lines,
                                    const        int     step,
                                    const        int     line_stride,
                                    const        int     n_comp,
                                    const        int     seg_len,
                                    const        int     n_segs,
                                    const        int     warmup,
                                    const        float4  b_hi,
                                    const        float4  b_lo,
                                    const        float4  mb0,
                                    const        float4  mb1,
                                    const        float4  mb2,
                                    const        int     clamp_edges,
                                    const        float4  edge4)
{
  const int gid  = get_global_id (0);
  const int c    = gid % n_comp;
  const int line = (gid / n_comp) % n_lines;
  const int seg  = gid / (n_comp * n_lines);

  if (seg >= n_segs)
    return;

  const global float *src   = src_buf + line * line_stride * n_comp + c;
        global float *dst   = dst_buf + line * line_stride * n_comp + c;
        global float *state = state_buf + (line * n_comp + c) * 6;

  const int step_c = step * n_comp;
  const int start  = seg * seg_len;
  const int end    = min (start + seg_len, len);
  const float edge = c == 0 ? edge4.x :
                     c == 1 ? edge4.y :
                     c == 2 ? edge4.z : edge4.w;
  float h0, h1, h2, h3;
  float l0, l1, l2, l3;
  int   i;

  i = start > warmup ? start - warmup : 0;

  if (i == 0)
    h1 = h2 = h3 = clamp_edges ? src[0] : edge;
  else
    h1 = h2 = h3 = src[i * step_c];

  l1 = l2 = l3 = 0.0f;

  for (; i < end; i++)
    {
      iir_step (src[i * step_c], &h0, &l0, h1, l1, h2, l2, h3, l3, b_hi, b_lo);

      if (i >= start)
        dst[i * step_c] = h0 + l0;

      h3 = h2; l3 = l2;
      h2 = h1; l2 = l1;
      h1 = h0; l1 = l0;
    }

  if (seg == n_segs - 1)
    {
      /* Right boundary, as in iir_young_causal () */
      const float uplus = clamp_edges ? src[(len - 1) * step_c] : edge;
      const float u0    = (h1 - uplus) + l1;
      const float du1   = (h2 - h1) + (l2 - l1);
      const float du2   = ((h3 - h2) + (l3 - l2)) - du1;
      const float t0    = mb0.x * u0 + mb0.y * du1 + mb0.z * du2;
      const float t1    = mb1.x * u0 + mb1.y * du1 + mb1.z * du2;
      const float t2    = mb2.x * u0 + mb2.y * du1 + mb2.z * du2;

      h1 = two_sum (t0, uplus, &l1);
      h2 = h1; l2 = l1;
      h3 = h1; l3 = l1;
      dd_add (&h2, &l2, t1);
      dd_add (&h3, &l3, t2);

      state[0] = h1; state[1] = h2; state[2] = h3;
      state[3] = l1; state[4] = l2; state[5] = l3;
    }
}

__kernel void gblur_mem_iir_anticausal (const global float *src_buf,
                                              global float *dst_buf,
                                        const global float *state_buf,
                                        const        int     len,
                                        const        int     n_lines,
                                        const        int     step,
                                        const        int     line_stride,
                                        const        int     n_comp,
                                        const        int     seg_len,
                                        const        int     n_segs,
                                        const        int     warmup,
                                        const        float4  b_hi,
                                        const        float4  b_lo)
{
  const int gid  = get_global_id (0);
  const int c    = gid % n_comp;
  const int line = (gid / n_comp) % n_lines;
  const int seg  = gid / (n_comp * n_lines);

  if (seg >= n_segs)
    return;

  const global float *src   = src_buf + line * line_stride * n_comp + c;
        global float *dst   = dst_buf + line * line_stride * n_comp + c;
  const global float *state = state_buf + (line * n_comp + c) * 6;

  const int step_c = step * n_comp;
  const int start  = seg * seg_len;
  const int end    = min (start + seg_len, len);
  float h0, h1, h2, h3;
  float l0, l1, l2, l3;
  int   i;

  if (end - 1 + warmup >= len - 1)
    {
      i  = len - 1;
      h1 = state[0]; h2 = state[1]; h3 = state[2];
      l1 = state[3]; l2 = state[4]; l3 = state[5];
    }
  else
    {
      i  = end - 1 + warmup;
      h1 = h2 = h3 = src[i * step_c];
      l1 = l2 = l3 = 0.0f;
    }

  for (; i >= start; i--)
    {
      iir_step (src[i * step_c], &h0, &l0, h1, l1, h2, l2, h3, l3, b_hi, b_lo);

      if (i < end)
        dst[i * step_c] = h0 + l0;

      h3 = h2; l3 = l2;
      h2 = h1; l2 = l1;
      h1 = h0; l1 = l0;
    }
}
