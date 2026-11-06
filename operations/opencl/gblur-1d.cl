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
 *
 * Copyright 2013 Victor Oliveira <victormatheus@gmail.com>
 * Copyright 2013 Téo Mazars      <teomazars@gmail.com>
 */

__kernel void fir_ver_blur(const global float4 *src_buf,
                                 global float4 *dst_buf,
                           const global float  *cmatrix,
                           const        int     clen)
{
    const int gidx          = get_global_id (0);
    const int gidy          = get_global_id (1);
    const int src_rowstride = get_global_size (0);
    const int dst_rowstride = get_global_size (0);

    const int half_clen     = clen / 2;

    const int src_offset    = gidx + (gidy + half_clen) * src_rowstride;
    const int dst_offset    = gidx +  gidy              * dst_rowstride;

    const int src_start_ind = src_offset - half_clen * src_rowstride;

    float4 v = 0.0f;

    for (int i = 0; i < clen; i++)
      {
        v += src_buf[src_start_ind + i * src_rowstride] * cmatrix[i];
      }

    dst_buf[dst_offset] = v;
}


__kernel void fir_hor_blur(const global float4 *src_buf,
                                 global float4 *dst_buf,
                           const global float  *cmatrix,
                           const        int     clen)
{
    const int gidx          = get_global_id (0);
    const int gidy          = get_global_id (1);
    const int src_rowstride = get_global_size (0) + clen - 1; /*== 2*(clen/2) */
    const int dst_rowstride = get_global_size (0);

    const int half_clen     = clen / 2;

    const int src_offset    = gidx + gidy * src_rowstride + half_clen;
    const int dst_offset    = gidx + gidy * dst_rowstride;

    const int src_start_ind = src_offset - half_clen;

    float4 v = 0.0f;

    for (int i = 0; i < clen; i++)
      {
        v += src_buf[src_start_ind + i] * cmatrix[i];
      }

    dst_buf[dst_offset] = v;
}

/* Young / van Vliet recursive gaussian, one work-item per line.
 *
 * This follows iir_young_blur_1D_rgbA () in gblur-1d.c: a causal pass
 * started from three copies of iminus, the Triggs / Sdika right boundary
 * fix using uplus, then an anti-causal pass over the causal results.
 *
 * Buffers hold n_comp float components per pixel (1 to 4, any of the
 * formats gblur-1d works in) and every work-item filters one component
 * of one line. A line is len pixels, consecutive pixels are step pixels
 * apart and consecutive lines are line_stride pixels apart, so the same
 * kernels handle rows (step = 1) and columns (step = width).
 *
 * The CPU code runs the recursion in double. For large std-dev the poles
 * get very close to 1 (b0 is ~4e-6 at std-dev 100), and plain float
 * evaluation of w = b0 x + b1 w1 + b2 w2 + b3 w3 drifts by tens of 8-bit
 * levels. Without fp64 we keep the accuracy by:
 *
 *  - evaluating the recursion in increment form, using b0+b1+b2+b3 == 1:
 *      w = w1 + b0 (x - w1) + b2 (w2 - w1) + b3 (w3 - w1)
 *    with the last two terms rewritten on first and second differences
 *    (see iir_step ()), so only small differences get multiplied, with
 *    the coefficients split into a float hi and lo part;
 *  - keeping the state as an unevaluated float sum hi + lo, updated with
 *    an error-free TwoSum, so rounding the state does not accumulate;
 *  - writing the boundary matrix product in terms of first and second
 *    differences (rows of the matrix are large and nearly cancel).
 */

/* s + e == a + b exactly */
static float
two_sum (const float  a,
         const float  b,
         float       *e)
{
  const float s  = a + b;
  const float bb = s - a;

  *e = (a - (s - bb)) + (b - bb);

  return s;
}

/* (*h, *l) += d */
static void
dd_add (float       *h,
        float       *l,
        const float  d)
{
  float e;
  const float s = two_sum (*h, d, &e);

  e += *l;
  *h = s + e;
  *l = e - (*h - s);
}

/* One recursion step: returns w0 = (*h0, *l0) from x and w1, w2, w3 */
static void
iir_step (const float  x,
          float       *h0,
          float       *l0,
          const float  h1,
          const float  l1,
          const float  h2,
          const float  l2,
          const float  h3,
          const float  l3,
          const float4  b_hi,
          const float4  b_lo)
{
  const float dx  = (x - h1) - l1;
  const float d2  = (h2 - h1) + (l2 - l1);
  const float dd3 = ((h3 - h2) - (h2 - h1)) + ((l3 - l2) - (l2 - l1));
  float d;

  /* b0 (x - w1) + b2 (w2 - w1) + b3 (w3 - w1) written as
   * b0 (x - w1) + (b2 + 2 b3) (w2 - w1) + b3 (w3 - 2 w2 + w1),
   * b_hi / b_lo hold (b0, unused, b2 + 2 b3, b3)
   */
  d  = b_hi.x * dx  + b_lo.x * dx;
  d += b_hi.z * d2  + b_lo.z * d2;
  d += b_hi.w * dd3 + b_lo.w * dd3;

  *h0 = h1;
  *l0 = l1;
  dd_add (h0, l0, d);
}

/* Lines longer than the OpenCL iteration size are filtered in segments:
 * the causal kernel runs over the segments in order and the anti-causal
 * kernel in reverse order, carrying the recursion state of every line in
 * state_buf (h1, h2, h3, l1, l2, l3 per line) from one segment to the next.
 * A single segment is just first == last == 1.
 */
__kernel void iir_young_causal (const global float *src_buf,
                                      global float *dst_buf,
                                      global float *state_buf,
                                const        int     len,
                                const        int     n_lines,
                                const        int     step,
                                const        int     line_stride,
                                const        int     line_offset,
                                const        int     n_comp,
                                const        int     first,
                                const        int     last,
                                const        float4  b_hi,
                                const        float4  b_lo,
                                const        float4  mb0,
                                const        float4  mb1,
                                const        float4  mb2,
                                const        int     clamp_edges,
                                const        float4  edge4)
{
  const int gid  = get_global_id (0);
  const int line = gid / n_comp;
  const int c    = gid % n_comp;

  if (line >= n_lines)
    return;

  const global float *src   = src_buf + line * line_stride * n_comp + c;
        global float *dst   = dst_buf + line * line_stride * n_comp + c;
        global float *state = state_buf +
                              ((line_offset + line) * n_comp + c) * 6;

  const int step_c = step * n_comp;
  const float edge = c == 0 ? edge4.x :
                     c == 1 ? edge4.y :
                     c == 2 ? edge4.z : edge4.w;
  float h0, h1, h2, h3;
  float l0, l1, l2, l3;
  int   i;

  if (first)
    {
      h1 = h2 = h3 = clamp_edges ? src[0] : edge;
      l1 = l2 = l3 = 0.0f;
    }
  else
    {
      h1 = state[0]; h2 = state[1]; h3 = state[2];
      l1 = state[3]; l2 = state[4]; l3 = state[5];
    }

  for (i = 0; i < len; i++)
    {
      iir_step (src[i * step_c], &h0, &l0, h1, l1, h2, l2, h3, l3, b_hi, b_lo);

      dst[i * step_c] = h0 + l0;

      h3 = h2; l3 = l2;
      h2 = h1; l2 = l1;
      h1 = h0; l1 = l0;
    }

  if (last)
    {
      /* Right boundary. With u_k = causal[len - 1 - k] - uplus the CPU
       * code computes the three anti-causal start values
       *   t_i = uplus + sum_k m[i][k] u_k.
       * m has huge, nearly cancelling entries for large std-dev, and the
       * anti-causal pass amplifies any inconsistency between the three
       * start values by orders of magnitude. So t_0 is computed from u_0,
       * the first difference u_1 - u_0 and the second difference
       * u_2 - 2 u_1 + u_0 (mb0 = row sum, m[0][1] + 2 m[0][2], m[0][2]),
       * and t_1, t_2 as small offsets from t_0 (mb1, mb2: the same terms
       * for row i minus those of row 0).
       */
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
    }

  state[0] = h1; state[1] = h2; state[2] = h3;
  state[3] = l1; state[4] = l2; state[5] = l3;
}

__kernel void iir_young_anticausal (const global float *src_buf,
                                          global float *dst_buf,
                                          global float *state_buf,
                                    const        int     len,
                                    const        int     n_lines,
                                    const        int     step,
                                    const        int     line_stride,
                                    const        int     line_offset,
                                    const        int     n_comp,
                                    const        float4  b_hi,
                                    const        float4  b_lo)
{
  const int gid  = get_global_id (0);
  const int line = gid / n_comp;
  const int c    = gid % n_comp;

  if (line >= n_lines)
    return;

  const global float *src   = src_buf + line * line_stride * n_comp + c;
        global float *dst   = dst_buf + line * line_stride * n_comp + c;
        global float *state = state_buf +
                              ((line_offset + line) * n_comp + c) * 6;

  const int step_c = step * n_comp;
  float h0, h1, h2, h3;
  float l0, l1, l2, l3;
  int   i;

  h1 = state[0]; h2 = state[1]; h3 = state[2];
  l1 = state[3]; l2 = state[4]; l3 = state[5];

  for (i = len - 1; i >= 0; i--)
    {
      iir_step (src[i * step_c], &h0, &l0, h1, l1, h2, l2, h3, l3, b_hi, b_lo);

      dst[i * step_c] = h0 + l0;

      h3 = h2; l3 = l2;
      h2 = h1; l2 = l1;
      h1 = h0; l1 = l0;
    }

  state[0] = h1; state[1] = h2; state[2] = h3;
  state[3] = l1; state[4] = l2; state[5] = l3;
}
