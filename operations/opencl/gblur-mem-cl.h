/* This file is part of GEGL
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

/* Gaussian blur of an OpenCL memory object, for operations that run
 * gegl:gaussian-blur on intermediate images and want to keep those
 * images on the GPU.
 *
 * gblur_mem_cl_blur () gives the same result as gegl:gaussian-blur with
 * the "auto" filter and clip-extent set, applied to a buffer whose extent
 * is the whole image: a horizontal and then a vertical gblur-1d pass,
 * each using the FIR filter for std-dev < 1.0 and the Young / van Vliet
 * IIR filter otherwise, with the "clamp" or "none" abyss policy. The IIR
 * passes use the recursion of gblur-1d.cl, with the constants computed as
 * in gblur-1d.c, but filter segments of every line in parallel (see
 * gblur-mem.cl).
 */

#ifndef GBLUR_MEM_CL_H
#define GBLUR_MEM_CL_H

#include "opencl/gegl-cl.h"
#include "gegl-buffer-cl-iterator.h"

#include "opencl/gblur-1d.cl.h"
#include "opencl/gblur-mem.cl.h"

static GeglClRunData *gblur_mem_cl_data = NULL;

static gboolean
gblur_mem_cl_build (void)
{
  static gchar *source = NULL;

  if (! gblur_mem_cl_data)
    {
      const char *kernel_name[] = { "gblur_mem_iir_causal",
                                    "gblur_mem_iir_anticausal",
                                    "gblur_mem_fir_1",
                                    "gblur_mem_fir_2",
                                    "gblur_mem_fir_4",
                                    "gblur_mem_put_1",
                                    "gblur_mem_put_2",
                                    "gblur_mem_put_4",
                                    NULL };

      if (! source)
        source = g_strconcat (gblur_1d_cl_source, "\n",
                              gblur_mem_cl_source, NULL);

      gblur_mem_cl_data = gegl_cl_compile_and_build (source, kernel_name);
    }

  return gblur_mem_cl_data != NULL;
}

/* Same as in gblur-1d.c */
static void
gblur_mem_iir_young_find_constants (gfloat   sigma,
                                    gdouble *b,
                                    gdouble (*m)[3])
{
  const gdouble K1 = 2.44413;
  const gdouble K2 = 1.4281;
  const gdouble K3 = 0.422205;
  const gdouble q = sigma >= 2.5 ?
                    0.98711 * sigma - 0.96330 :
                    3.97156 - 4.14554 * sqrt (1 - 0.26891 * sigma);

  const gdouble b0 = 1.57825 + q*(K1 + q*(    K2 + q *     K3));
  const gdouble b1 =           q*(K1 + q*(2 * K2 + q * 3 * K3));
  const gdouble b2 =     (-K2 * q * q) + (-K3 * 3 * q * q * q);
  const gdouble b3 =           q*      q*          q *     K3;

  const gdouble a1 = b1 / b0;
  const gdouble a2 = b2 / b0;
  const gdouble a3 = b3 / b0;

  const gdouble c  = 1. / ((1+a1-a2+a3) * (1+a2+(a1-a3)*a3));

  m[0][0] = c * (-a3*(a1+a3)-a2 + 1);
  m[0][1] = c * (a3+a1)*(a2+a3*a1);
  m[0][2] = c * a3*(a1+a3*a2);

  m[1][0] = c * (a1+a3*a2);
  m[1][1] = c * (1-a2)*(a2+a3*a1);
  m[1][2] = c * a3*(1-a3*a1-a3*a3-a2);

  m[2][0] = c * (a3*a1+a2+a1*a1-a2*a2);
  m[2][1] = c * (a1*a2+a3*a2*a2-a1*a3*a3-a3*a3*a3-a3*a2+a3);
  m[2][2] = c * a3*(a1+a3*a2);

  b[0] = 1. - (b1 + b2 + b3) / b0;
  b[1] = a1;
  b[2] = a2;
  b[3] = a3;
}

/* Same as in gblur-1d.c */
static gint
gblur_mem_fir_gen_convolve_matrix (gfloat   sigma,
                                   gfloat **cmatrix)
{
  gint    clen;
  gfloat *cmatrix_p;

  clen = sigma > GEGL_FLOAT_EPSILON ? ceil (sigma * 6.5) : 1;
  clen = clen + ((clen + 1) % 2);

  *cmatrix  = g_new (gfloat, clen);
  cmatrix_p = *cmatrix;

  if (clen == 1)
    {
      cmatrix_p [0] = 1;
    }
  else
    {
      gint    i;
      gdouble sum = 0;
      gint    half_clen = clen / 2;

      for (i = 0; i < clen; i++)
        {
          gfloat x = i - half_clen;

          cmatrix_p [i] = (1.0 / (sigma * sqrt (2.0 * G_PI))) *
                          exp (-(x * x) / (2.0 * sigma * sigma));
          sum += cmatrix_p [i];
        }

      for (i = 0; i < clen; i++)
        {
          cmatrix_p [i] /= sum;
        }
    }

  return clen;
}

/* The number of pixels after which the effect of the start values of
 * the IIR recursion has decayed below float precision.
 */
static gint
gblur_mem_iir_warmup (const gdouble *b)
{
  gint k, n = 0;

  for (k = 0; k < 3; k++)
    {
      gdouble w[3] = { 0.0, 0.0, 0.0 };
      gint    i;

      w[k] = 1.0;

      for (i = 1; i < 100000; i++)
        {
          const gdouble w0 = b[1] * w[0] + b[2] * w[1] + b[3] * w[2];

          w[2] = w[1];
          w[1] = w[0];
          w[0] = w0;

          if (fabs (w[0]) < 1e-8 && fabs (w[1]) < 1e-8 && fabs (w[2]) < 1e-8)
            break;
        }

      n = MAX (n, i);
    }

  return n;
}

/* One gblur-1d pass from src into dst, using tmp for the IIR filter;
 * the three must differ.
 */
static gboolean
gblur_mem_cl_pass (cl_mem   src,
                   cl_mem   dst,
                   cl_mem   tmp,
                   gint     width,
                   gint     height,
                   gint     n_comp,
                   gfloat   std_dev,
                   gboolean horizontal,
                   gboolean clamp_edges)
{
  cl_int cl_err = CL_SUCCESS;
  cl_int horizontal_i  = horizontal;
  cl_int clamp_edges_i = clamp_edges;
  cl_int width_i       = width;
  cl_int height_i      = height;
  cl_int n_comp_i      = n_comp;

  if (std_dev < 1.0)
    {
      /* FIR */
      cl_kernel kernel;
      cl_mem    cl_cmatrix = NULL;
      gfloat   *cmatrix;
      cl_int    clen;
      size_t    global_ws[2];

      if (n_comp == 3)
        return FALSE;

      kernel = gblur_mem_cl_data->kernel[n_comp == 4 ? 4 : n_comp + 1];
      clen   = gblur_mem_fir_gen_convolve_matrix (std_dev, &cmatrix);

      cl_cmatrix = gegl_clCreateBuffer (gegl_cl_get_context (),
                                        CL_MEM_COPY_HOST_PTR |
                                        CL_MEM_READ_ONLY,
                                        clen * sizeof (cl_float), cmatrix,
                                        &cl_err);
      g_free (cmatrix);
      if (cl_err != CL_SUCCESS)
        return FALSE;

      cl_err = gegl_cl_set_kernel_args (kernel,
                                        sizeof (cl_mem), &src,
                                        sizeof (cl_mem), &dst,
                                        sizeof (cl_mem), &cl_cmatrix,
                                        sizeof (cl_int), &clen,
                                        sizeof (cl_int), &width_i,
                                        sizeof (cl_int), &height_i,
                                        sizeof (cl_int), &horizontal_i,
                                        sizeof (cl_int), &clamp_edges_i,
                                        NULL);

      if (cl_err == CL_SUCCESS)
        {
          global_ws[0] = width;
          global_ws[1] = height;

          cl_err = gegl_clEnqueueNDRangeKernel (gegl_cl_get_command_queue (),
                                                kernel,
                                                2, NULL, global_ws, NULL,
                                                0, NULL, NULL);
        }

      gegl_clReleaseMemObject (cl_cmatrix);

      return cl_err == CL_SUCCESS;
    }
  else
    {
      /* IIR, see iir_cl_process () in gblur-1d.c */
      gdouble   b[4], m[3][3], bc[4], mb[3][3];
      cl_float4 b_hi, b_lo, mbf[3], edge;
      cl_mem    state;
      cl_int    len, n_lines, step, line_stride;
      cl_int    seg_len, n_segs, warmup;
      size_t    global_ws;
      gint      k, j;

      gblur_mem_iir_young_find_constants (std_dev, b, m);

      bc[0] = b[0];
      bc[1] = b[1];
      bc[2] = b[2] + 2.0 * b[3];
      bc[3] = b[3];

      for (k = 0; k < 4; k++)
        {
          b_hi.s[k] = bc[k];
          b_lo.s[k] = bc[k] - (gdouble) b_hi.s[k];
          edge.s[k] = 0.0f;
        }

      for (k = 0; k < 3; k++)
        {
          mb[k][0] = m[k][0] + m[k][1] + m[k][2];
          mb[k][1] = m[k][1] + 2.0 * m[k][2];
          mb[k][2] = m[k][2];
        }

      for (k = 0; k < 3; k++)
        {
          for (j = 0; j < 3; j++)
            mbf[k].s[j] = k == 0 ? mb[0][j] : mb[k][j] - mb[0][j];

          mbf[k].s[3] = 0.0f;
        }

      if (horizontal)
        {
          len         = width;
          n_lines     = height;
          step        = 1;
          line_stride = width;
        }
      else
        {
          len         = height;
          n_lines     = width;
          step        = width;
          line_stride = 1;
        }

      /* Enough work-items to fill a GPU, with segments of at least the
       * warm-up length
       */
      warmup  = gblur_mem_iir_warmup (b);
      n_segs  = (32768 + n_lines * n_comp - 1) / (n_lines * n_comp);
      n_segs  = CLAMP (n_segs, 1, len / MAX (warmup, 32));
      n_segs  = MAX (n_segs, 1);
      seg_len = (len + n_segs - 1) / n_segs;
      n_segs  = (len + seg_len - 1) / seg_len;

      state = gegl_clCreateBuffer (gegl_cl_get_context (),
                                   CL_MEM_READ_WRITE,
                                   (size_t) n_lines * n_comp * 6 *
                                   sizeof (cl_float),
                                   NULL, &cl_err);
      if (cl_err != CL_SUCCESS)
        return FALSE;

      global_ws = (size_t) n_segs * n_lines * n_comp;

      cl_err = gegl_cl_set_kernel_args (gblur_mem_cl_data->kernel[0],
                                        sizeof (cl_mem),    &src,
                                        sizeof (cl_mem),    &tmp,
                                        sizeof (cl_mem),    &state,
                                        sizeof (cl_int),    &len,
                                        sizeof (cl_int),    &n_lines,
                                        sizeof (cl_int),    &step,
                                        sizeof (cl_int),    &line_stride,
                                        sizeof (cl_int),    &n_comp_i,
                                        sizeof (cl_int),    &seg_len,
                                        sizeof (cl_int),    &n_segs,
                                        sizeof (cl_int),    &warmup,
                                        sizeof (cl_float4), &b_hi,
                                        sizeof (cl_float4), &b_lo,
                                        sizeof (cl_float4), &mbf[0],
                                        sizeof (cl_float4), &mbf[1],
                                        sizeof (cl_float4), &mbf[2],
                                        sizeof (cl_int),    &clamp_edges_i,
                                        sizeof (cl_float4), &edge,
                                        NULL);
      if (cl_err == CL_SUCCESS)
        cl_err = gegl_clEnqueueNDRangeKernel (gegl_cl_get_command_queue (),
                                              gblur_mem_cl_data->kernel[0],
                                              1, NULL, &global_ws, NULL,
                                              0, NULL, NULL);

      if (cl_err == CL_SUCCESS)
        cl_err = gegl_cl_set_kernel_args (gblur_mem_cl_data->kernel[1],
                                          sizeof (cl_mem),    &tmp,
                                          sizeof (cl_mem),    &dst,
                                          sizeof (cl_mem),    &state,
                                          sizeof (cl_int),    &len,
                                          sizeof (cl_int),    &n_lines,
                                          sizeof (cl_int),    &step,
                                          sizeof (cl_int),    &line_stride,
                                          sizeof (cl_int),    &n_comp_i,
                                          sizeof (cl_int),    &seg_len,
                                          sizeof (cl_int),    &n_segs,
                                          sizeof (cl_int),    &warmup,
                                          sizeof (cl_float4), &b_hi,
                                          sizeof (cl_float4), &b_lo,
                                          NULL);
      if (cl_err == CL_SUCCESS)
        cl_err = gegl_clEnqueueNDRangeKernel (gegl_cl_get_command_queue (),
                                              gblur_mem_cl_data->kernel[1],
                                              1, NULL, &global_ws, NULL,
                                              0, NULL, NULL);

      gegl_clReleaseMemObject (state);

      return cl_err == CL_SUCCESS;
    }
}

/* Blurs the width x height image of n_comp (1 to 4) floats per pixel in
 * src into dst, which may be the same memory object, like
 * gegl:gaussian-blur with std-dev-x = std-dev-y = std_dev. With
 * clamp_edges, pixels outside the image are the nearest edge pixel,
 * otherwise they are zero.
 */
static gboolean
gblur_mem_cl_blur (cl_mem   src,
                   cl_mem   dst,
                   gint     width,
                   gint     height,
                   gint     n_comp,
                   gdouble  std_dev,
                   gboolean clamp_edges)
{
  cl_int   cl_err = CL_SUCCESS;
  cl_mem   tmp1, tmp2;
  gboolean ok;
  size_t   size = (size_t) width * height * n_comp * sizeof (cl_float);

  if (n_comp < 1 || n_comp > 4 || width <= 0 || height <= 0)
    return FALSE;

  if (! gblur_mem_cl_build ())
    return FALSE;

  tmp1 = gegl_clCreateBuffer (gegl_cl_get_context (), CL_MEM_READ_WRITE,
                              size, NULL, &cl_err);
  if (cl_err != CL_SUCCESS)
    return FALSE;

  tmp2 = gegl_clCreateBuffer (gegl_cl_get_context (), CL_MEM_READ_WRITE,
                              size, NULL, &cl_err);
  if (cl_err != CL_SUCCESS)
    {
      gegl_clReleaseMemObject (tmp1);
      return FALSE;
    }

  /* gblur-1d takes the std-dev as a float */
  ok = gblur_mem_cl_pass (src, tmp2, tmp1, width, height, n_comp,
                          (gfloat) std_dev, TRUE, clamp_edges) &&
       gblur_mem_cl_pass (tmp2, dst, tmp1, width, height, n_comp,
                          (gfloat) std_dev, FALSE, clamp_edges);

  gegl_clReleaseMemObject (tmp2);
  gegl_clReleaseMemObject (tmp1);

  return ok;
}

/* Reads rect of buffer in format, which must have 1, 2 or 4 float
 * components, into dst as a dense rect->width x rect->height image. With
 * premultiply, the color components are multiplied with the alpha, the
 * last component, as a conversion to the associated alpha format does.
 */
static gboolean
gblur_mem_cl_read (GeglBuffer          *buffer,
                   const GeglRectangle *rect,
                   const Babl          *format,
                   cl_mem               dst,
                   gboolean             premultiply)
{
  const gint            n_comp = babl_format_get_n_components (format);
  const gint            bpp    = babl_format_get_bytes_per_pixel (format);
  GeglBufferClIterator *i;
  cl_kernel             kernel;
  gboolean              err = FALSE;
  cl_int                cl_err;
  cl_int                premultiply_i = premultiply;
  cl_int                stride = rect->width;

  if (n_comp == 3 || n_comp > 4 || bpp != n_comp * sizeof (gfloat))
    return FALSE;

  if (! gblur_mem_cl_build ())
    return FALSE;

  kernel = gblur_mem_cl_data->kernel[n_comp == 4 ? 7 : n_comp + 4];

  i = gegl_buffer_cl_iterator_new (buffer, rect, format, GEGL_CL_BUFFER_READ);

  while (gegl_buffer_cl_iterator_next (i, &err) && ! err)
    {
      const GeglRectangle *roi = &i->roi[0];
      const size_t global_ws[2] = { roi->width, roi->height };
      cl_int       chunk_width  = roi->width;
      cl_int       x0           = roi->x - rect->x;
      cl_int       y0           = roi->y - rect->y;

      if (premultiply)
        {
          cl_err = gegl_cl_set_kernel_args (kernel,
                                            sizeof (cl_mem), &i->tex[0],
                                            sizeof (cl_mem), &dst,
                                            sizeof (cl_int), &chunk_width,
                                            sizeof (cl_int), &x0,
                                            sizeof (cl_int), &y0,
                                            sizeof (cl_int), &stride,
                                            sizeof (cl_int), &premultiply_i,
                                            NULL);
          if (cl_err == CL_SUCCESS)
            cl_err = gegl_clEnqueueNDRangeKernel (gegl_cl_get_command_queue (),
                                                  kernel, 2,
                                                  NULL, global_ws, NULL,
                                                  0, NULL, NULL);
        }
      else
        {
          const size_t src_origin[3] = { 0, 0, 0 };
          const size_t dst_origin[3] = { (size_t) x0 * bpp, (size_t) y0, 0 };
          const size_t region[3]     = { (size_t) roi->width * bpp,
                                         (size_t) roi->height, 1 };

          cl_err = gegl_clEnqueueCopyBufferRect (gegl_cl_get_command_queue (),
                                                 i->tex[0], dst,
                                                 src_origin, dst_origin,
                                                 region,
                                                 (size_t) roi->width * bpp, 0,
                                                 (size_t) stride * bpp, 0,
                                                 0, NULL, NULL);
        }
      if (cl_err != CL_SUCCESS)
        {
          gegl_buffer_cl_iterator_stop (i);
          return FALSE;
        }
    }

  return ! err;
}

#endif /* GBLUR_MEM_CL_H */
