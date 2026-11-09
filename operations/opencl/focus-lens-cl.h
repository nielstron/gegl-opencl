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

#ifndef __FOCUS_LENS_CL_H__
#define __FOCUS_LENS_CL_H__

#include <string.h>

#include "opencl/gegl-cl.h"
#include "gegl-buffer-cl-iterator.h"

#include "opencl/focus-blur-mask.h"
#include "opencl/focus-lens-blur.cl.h"

static gboolean
focus_lens_cl_tile (GeglClRunData       *cl_data,
                    GeglBuffer         *input,
                    GeglBuffer         *output,
                    const GeglRectangle *extent,
                    const GeglRectangle *tile,
                    const Babl         *format,
                    const FocusBlurMask *mask,
                    cl_mem              spans,
                    cl_mem              fallback,
                    gboolean            try_opaque,
                    cl_int              iradius,
                    cl_float            radius,
                    const cl_float4    *luminance,
                    cl_float            threshold_low,
                    cl_float            threshold_high,
                    cl_float            factor,
                    cl_float            highlight_max)
{
  GeglBufferClIterator *iter;
  gint                  left = MIN (iradius, tile->x - extent->x);
  gint                  right = MIN (iradius, extent->x + extent->width -
                                              tile->x - tile->width);
  gint                  top = MIN (iradius, tile->y - extent->y);
  gint                  bottom = MIN (iradius, extent->y + extent->height -
                                               tile->y - tile->height);
  gint                  read, aux;
  gboolean              failed;
  cl_int                cl_err;
  cl_mem                pixels = NULL;
  cl_mem                disc = NULL;
  cl_mem                delta = NULL;
  cl_mem                delta_w = NULL;
  cl_mem                opacity = NULL;
  gboolean              opaque = FALSE;
  cl_kernel             prepare, blur;
  cl_int                strip_width = CLAMP (2 * iradius, 64, 256);
  const GeglRectangle  *src_rect;
  cl_int4               source;
  size_t                global_ws[2];

  iter = gegl_buffer_cl_iterator_new (output, tile, format,
                                     GEGL_CL_BUFFER_WRITE);
  read = gegl_buffer_cl_iterator_add_2 (iter, input, tile, format,
                                        GEGL_CL_BUFFER_READ,
                                        left, right, top, bottom,
                                        GEGL_ABYSS_NONE);
  aux = gegl_buffer_cl_iterator_add_aux (iter, tile, babl_format ("Y float"),
                                          left, right, top, bottom);

  /* The tile is at most one iterator chunk. */
  if (!gegl_buffer_cl_iterator_next (iter, &failed))
    return FALSE;

  src_rect = &iter->roi[read];

  if (try_opaque)
    {
      cl_int    count = iter->size[read];
      size_t    checks = (count + 63) / 64;
      guchar   *flags = g_alloca (checks);
      cl_kernel check = cl_data->kernel[3];

      opacity = gegl_cl_mem_pool_get (CL_MEM_READ_WRITE, checks, &cl_err);
      CL_CHECK;

      cl_err = gegl_cl_set_kernel_args (check,
                                        sizeof (cl_mem), &iter->tex[read],
                                        sizeof (cl_mem), &opacity,
                                        sizeof (cl_int), &count,
                                        NULL);
      CL_CHECK;
      cl_err = gegl_clEnqueueNDRangeKernel (gegl_cl_get_command_queue (),
                                            check, 1, NULL, &checks, NULL,
                                            0, NULL, NULL);
      CL_CHECK;
      cl_err = gegl_clEnqueueReadBuffer (gegl_cl_get_command_queue (),
                                         opacity, CL_TRUE, 0, checks, flags,
                                         0, NULL, NULL);
      CL_CHECK;

      opaque = memchr (flags, 1, checks) == NULL;
    }

  prepare = cl_data->kernel[opaque ? 4 : fallback ? 2 : 0];
  blur    = cl_data->kernel[opaque ? 5 : 1];

  pixels = gegl_cl_mem_pool_get (CL_MEM_READ_WRITE,
                                 iter->size[read] * sizeof (cl_float4), &cl_err);
  CL_CHECK;
  disc = gegl_cl_mem_pool_get (CL_MEM_READ_WRITE,
                               iter->size[read] * (opaque ? sizeof (cl_float) : sizeof (cl_float2)), &cl_err);
  CL_CHECK;
  delta = gegl_cl_mem_pool_get (CL_MEM_READ_WRITE,
                                (opaque ? 1 : 2) * iter->size[0] * sizeof (cl_float4), &cl_err);
  CL_CHECK;
  if (!opaque)
    {
      delta_w = gegl_cl_mem_pool_get (CL_MEM_READ_WRITE,
                                      iter->size[0] * sizeof (cl_float4), &cl_err);
      CL_CHECK;
    }

  if (!focus_blur_mask_enqueue (mask, src_rect, iter->tex[aux]))
    goto error;

  global_ws[0] = src_rect->width;
  global_ws[1] = src_rect->height;

  if (opaque)
    cl_err = gegl_cl_set_kernel_args (prepare,
                                     sizeof (cl_mem),   &iter->tex[read],
                                     sizeof (cl_mem),   &iter->tex[aux],
                                     sizeof (cl_mem),   &pixels,
                                     sizeof (cl_mem),   &disc,
                                     sizeof (cl_float), &radius,
                                     NULL);
  else
    cl_err = gegl_cl_set_kernel_args (prepare,
                                      sizeof (cl_mem),    &iter->tex[read],
                                      sizeof (cl_mem),    &iter->tex[aux],
                                      sizeof (cl_mem),    &pixels,
                                      sizeof (cl_mem),    &disc,
                                      sizeof (cl_float),  &radius,
                                      sizeof (cl_float4), luminance,
                                      sizeof (cl_float),  &threshold_low,
                                      sizeof (cl_float),  &threshold_high,
                                      sizeof (cl_float),  &factor,
                                      sizeof (cl_float),  &highlight_max,
                                      NULL);
  CL_CHECK;

  if (fallback)
    {
      cl_err = gegl_clSetKernelArg (prepare, 10, sizeof (cl_mem), &fallback);
      CL_CHECK;
    }

  cl_err = gegl_clEnqueueNDRangeKernel (gegl_cl_get_command_queue (),
                                        prepare, 2,
                                        NULL, global_ws, NULL, 0, NULL, NULL);
  CL_CHECK;

  if (fallback)
    {
      cl_int invalid;

      cl_err = gegl_clEnqueueReadBuffer (gegl_cl_get_command_queue (),
                                         fallback, CL_TRUE, 0, sizeof (invalid),
                                         &invalid, 0, NULL, NULL);
      CL_CHECK;

      if (invalid)
        goto error;
    }

  source.s[0] = src_rect->width;
  source.s[1] = src_rect->height;
  source.s[2] = left;
  source.s[3] = top;
  global_ws[0] = tile->height;
  global_ws[1] = (tile->width + strip_width - 1) / strip_width;

  cl_err = gegl_cl_set_kernel_args (blur,
                                    sizeof (cl_mem),  &pixels,
                                    sizeof (cl_mem),  &disc,
                                    sizeof (cl_mem),  &delta,
                                    sizeof (cl_mem),  &iter->tex[0],
                                    sizeof (cl_mem),  &spans,
                                    sizeof (cl_int4), &source,
                                    sizeof (cl_int),  &tile->width,
                                    sizeof (cl_int),  &iradius,
                                    sizeof (cl_int),  &strip_width,
                                    NULL);
  CL_CHECK;

  if (!opaque)
    {
      cl_err = gegl_clSetKernelArg (blur, 9, sizeof (cl_mem), &delta_w);
      CL_CHECK;
    }

  cl_err = gegl_clEnqueueNDRangeKernel (gegl_cl_get_command_queue (),
                                        blur, 2,
                                        NULL, global_ws, NULL, 0, NULL, NULL);
  CL_CHECK;

  gegl_buffer_cl_iterator_next (iter, &failed);
  goto cleanup;

error:
  gegl_buffer_cl_iterator_stop (iter);
  failed = TRUE;

cleanup:
  gegl_cl_mem_pool_put (opacity);
  gegl_cl_mem_pool_put (delta_w);
  gegl_cl_mem_pool_put (delta);
  gegl_cl_mem_pool_put (disc);
  gegl_cl_mem_pool_put (pixels);

  return !failed;
}

static gboolean
focus_lens_cl_process (GeglBuffer          *input,
                       GeglBuffer          *output,
                       const GeglRectangle *extent,
                       const GeglRectangle *result,
                       const Babl          *format,
                       const FocusBlurMask *mask,
                       gdouble              blur_radius,
                       gdouble              highlight_factor,
                       gdouble              threshold_low,
                       gdouble              threshold_high)
{
  static GeglClRunData *cl_data = NULL;
  const cl_float        radius = blur_radius;
  const cl_int          iradius = floorf (radius + 0.5f);
  const cl_float        factor = 10.0 * highlight_factor * G_LN2;
  const cl_float        highlight_max = expf (factor);
  const cl_float        low = threshold_low;
  const cl_float        high = threshold_high;
  const cl_float        range_floor = 0.0002f * MAX (1.0f, MAX (fabsf (low),
                                                               fabsf (high)));
  /* Keep the endpoint scratch small even on devices with large iterators. */
  const gint            tile_width = MIN (1024, gegl_cl_get_iter_width ());
  const gint            tile_height = MIN (256, gegl_cl_get_iter_height ());
  gdouble               lum[3];
  cl_float4             luminance;
  cl_int               *spans;
  cl_mem                spans_tex;
  cl_mem                fallback = NULL;
  cl_int                cl_err;
  const Babl           *source_format = gegl_buffer_get_format (input);
  gboolean              check = babl_format_is_palette (source_format);
  const gboolean        try_opaque = radius <= 32.0f && factor == 0.0f &&
    source_format == babl_format ("R'G'B'A u8");
  gboolean              success = TRUE;
  gint                  x, y, r;

  /* A narrow highlight range amplifies rounding differences between the
   * CPU luminance conversion and OpenCL dot product. Equal thresholds
   * can also produce NaNs that propagate across the CPU's row.
   */
  if (factor != 0.0f &&
      (!isfinite (low) || !isfinite (high) ||
       fabsf (high - low) < range_floor))
    return FALSE;

  if (!cl_data)
    {
      const char *kernels[] = { "focus_lens_prepare", "focus_lens_disc",
                                "focus_lens_prepare_checked", "focus_lens_check_opacity",
                                "focus_lens_prepare_opaque", "focus_lens_disc_opaque",
                                NULL };

#if defined (__aarch64__) || defined (__FMA__)
      gchar *source = g_strconcat ("#define FOCUS_LENS_USE_FMA 1\n",
                                   focus_lens_blur_cl_source, NULL);

      cl_data = gegl_cl_compile_and_build_optimized (source, kernels);
      g_free (source);
#else
      cl_data = gegl_cl_compile_and_build_optimized (focus_lens_blur_cl_source,
                                                     kernels);
#endif
    }

  if (!cl_data)
    return FALSE;

  babl_space_get_rgb_luminance (babl_format_get_space (format),
                                &lum[0], &lum[1], &lum[2]);
  luminance.s[0] = lum[0];
  luminance.s[1] = lum[1];
  luminance.s[2] = lum[2];
  luminance.s[3] = 0.0f;

  spans = g_new (cl_int, iradius + 1);
  for (r = 0; r <= iradius; r++)
    spans[r] = sqrtf ((radius + 0.5f) * (radius + 0.5f) - r * r);

  spans_tex = gegl_clCreateBuffer (gegl_cl_get_context (),
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   (iradius + 1) * sizeof (cl_int),
                                   spans, &cl_err);
  g_free (spans);

  if (cl_err != CL_SUCCESS)
    return FALSE;

  for (gint i = 0; !check && i < babl_format_get_n_components (source_format); i++)
    {
      const Babl *type = babl_format_get_type (source_format, i);

      check = type != babl_type ("u8") && type != babl_type ("u16") &&
              type != babl_type ("u32");
    }

  if (check)
    {
      cl_int zero = 0;

      fallback = gegl_clCreateBuffer (gegl_cl_get_context (),
                                      CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                      sizeof (zero), &zero, &cl_err);
      success = cl_err == CL_SUCCESS;
    }

  for (y = 0; success && y < result->height; y += tile_height)
    for (x = 0; success && x < result->width; x += tile_width)
      {
        GeglRectangle tile = {result->x + x, result->y + y,
                              MIN (tile_width, result->width - x),
                              MIN (tile_height, result->height - y)};

        success = focus_lens_cl_tile (cl_data, input, output, extent, &tile,
                                      format, mask, spans_tex, fallback, try_opaque,
                                      iradius, radius, &luminance, low, high,
                                      factor, highlight_max);
      }

  if (fallback)
    gegl_clReleaseMemObject (fallback);
  gegl_clReleaseMemObject (spans_tex);

  return success;
}

#endif
