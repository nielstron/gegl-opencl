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
 * Copyright (C) 2022 Thomas Manni <thomas.manni@free.fr>
 *
 * following the paper (without channels decorellation)
 *   Guoshen Yu, and Guillermo Sapiro
 *   DCT Image Denoising: a Simple and Effective Image Denoising Algorithm
 */

#include "config.h"
#include <glib/gi18n-lib.h>

#ifdef GEGL_PROPERTIES

enum_start (gegl_denoise_dct_patchsize)
    enum_value (GEGL_DENOISE_DCT_8X8,   "size8x8",   "8x8")
    enum_value (GEGL_DENOISE_DCT_16X16, "size16x16", "16x16")
enum_end (GeglDenoiseDctPatchsize)

property_enum (patch_size, _("Patch size"),
               GeglDenoiseDctPatchsize, gegl_denoise_dct_patchsize,
               GEGL_DENOISE_DCT_8X8)
    description (_("Size of patches used to denoise"))

property_double (sigma, _("Strength"), 5.0)
    description (_("Noise standard deviation"))
    value_range (1., 100.)

#else

#define GEGL_OP_FILTER
#define GEGL_OP_NAME     denoise_dct
#define GEGL_OP_C_SOURCE denoise-dct.cc
#include "gegl-op.h"
#include "dct-basis.inc"
#include "opencl/gegl-cl.h"
#include "gegl-buffer-cl-iterator.h"
#include "opencl/denoise-dct.cl.h"

/* Each scratch pixel contains up to 16 horizontal coefficients. Limit the
 * working set independently of the OpenCL iterator's default chunk size.
 */
#define CL_DCT_TILE_WIDTH  512
#define CL_DCT_TILE_HEIGHT 128
#define CL_DCT_BAND_HEIGHT 8

static GMutex cl_mutex;

static gboolean
cl_denoise_tile (GeglClRunData      *cl_data,
                 cl_mem             input,
                 cl_mem             output,
                 cl_mem             basis,
                 const GeglRectangle *input_rect,
                 const GeglRectangle *source_rect,
                 const GeglRectangle *output_rect,
                 cl_int             patch_size,
                 cl_float           threshold)
{
  cl_command_queue queue = gegl_cl_get_command_queue ();
  cl_mem           rows = NULL;
  cl_mem           columns = NULL;
  cl_int           cl_err;
  cl_int           input_offset = ((source_rect->y - input_rect->y) *
                                    input_rect->width +
                                    source_rect->x - input_rect->x);
  cl_int           output_x = output_rect->x - source_rect->x;
  cl_int           output_y = output_rect->y - source_rect->y;
  size_t           coeff_stride = (source_rect->width - patch_size + 1) * patch_size;
  size_t           rows_ws[] = { coeff_stride, (size_t) source_rect->height };
  size_t           bands = (output_rect->height + CL_DCT_BAND_HEIGHT - 1) /
                           CL_DCT_BAND_HEIGHT;
  size_t           columns_ws[] = { coeff_stride, patch_size * bands };
  size_t           output_ws[] = { (size_t) output_rect->width,
                                   (size_t) output_rect->height };
  size_t           group_width = patch_size == 8 ? 4 : 2;
  size_t           columns_local[] = { group_width, (size_t) patch_size };
  size_t           patch_rows = CL_DCT_BAND_HEIGHT + patch_size - 1;
  size_t           row_band_size = (patch_rows + patch_size - 1) *
                                    group_width * sizeof (cl_float4);
  size_t           coefficients_size = patch_rows * patch_size *
                                        group_width * sizeof (cl_float4);
  size_t           partial_size = CL_DCT_BAND_HEIGHT * patch_size *
                                   group_width * sizeof (cl_float4);

  if (group_width * patch_size > cl_data->work_group_size[1] ||
      row_band_size + coefficients_size + partial_size >
      gegl_cl_get_local_mem_size ())
    return FALSE;

  rows = gegl_cl_mem_pool_get (CL_MEM_READ_WRITE,
                               coeff_stride * source_rect->height *
                               sizeof (cl_float4), &cl_err);
  CL_CHECK;

  columns = gegl_cl_mem_pool_get (CL_MEM_READ_WRITE,
                                  coeff_stride * output_rect->height *
                                  sizeof (cl_float4), &cl_err);
  CL_CHECK;

  cl_err = gegl_cl_set_kernel_args (cl_data->kernel[0],
                                    sizeof (cl_mem), &input,
                                    sizeof (cl_mem), &rows,
                                    sizeof (cl_mem), &basis,
                                    sizeof (cl_int), &input_rect->width,
                                    sizeof (cl_int), &input_offset,
                                    sizeof (cl_int), &patch_size,
                                    NULL);
  CL_CHECK;

  cl_err = gegl_clEnqueueNDRangeKernel (queue, cl_data->kernel[0], 2, NULL,
                                        rows_ws, NULL, 0, NULL, NULL);
  CL_CHECK;

  cl_err = gegl_cl_set_kernel_args (cl_data->kernel[1],
                                    sizeof (cl_mem), &rows,
                                    sizeof (cl_mem), &columns,
                                    sizeof (cl_mem), &basis,
                                    row_band_size, NULL,
                                    coefficients_size, NULL,
                                    partial_size, NULL,
                                    sizeof (cl_int), &source_rect->height,
                                    sizeof (cl_int), &output_y,
                                    sizeof (cl_int), &output_rect->height,
                                    sizeof (cl_int), &patch_size,
                                    sizeof (cl_float), &threshold,
                                    NULL);
  CL_CHECK;

  cl_err = gegl_clEnqueueNDRangeKernel (queue, cl_data->kernel[1], 2, NULL,
                                        columns_ws, columns_local,
                                        0, NULL, NULL);
  CL_CHECK;

  rows_ws[1] = output_rect->height;
  cl_err = gegl_cl_set_kernel_args (cl_data->kernel[2],
                                    sizeof (cl_mem), &columns,
                                    sizeof (cl_mem), &rows,
                                    sizeof (cl_mem), &basis,
                                    sizeof (cl_int), &patch_size,
                                    NULL);
  CL_CHECK;

  cl_err = gegl_clEnqueueNDRangeKernel (queue, cl_data->kernel[2], 2, NULL,
                                        rows_ws, NULL, 0, NULL, NULL);
  CL_CHECK;

  cl_err = gegl_cl_set_kernel_args (cl_data->kernel[3],
                                    sizeof (cl_mem), &input,
                                    sizeof (cl_mem), &rows,
                                    sizeof (cl_mem), &output,
                                    sizeof (cl_int), &input_rect->width,
                                    sizeof (cl_int), &input_offset,
                                    sizeof (cl_int), &source_rect->width,
                                    sizeof (cl_int), &source_rect->height,
                                    sizeof (cl_int), &output_x,
                                    sizeof (cl_int), &output_y,
                                    sizeof (cl_int), &patch_size,
                                    NULL);
  CL_CHECK;

  cl_err = gegl_clEnqueueNDRangeKernel (queue, cl_data->kernel[3], 2, NULL,
                                        output_ws, NULL, 0, NULL, NULL);
  CL_CHECK;

  gegl_cl_mem_pool_put (columns);
  gegl_cl_mem_pool_put (rows);
  return TRUE;

error:
  if (columns)
    gegl_clReleaseMemObject (columns);
  if (rows)
    gegl_clReleaseMemObject (rows);
  return FALSE;
}

static gboolean
cl_denoise (GeglOperation       *operation,
            GeglBuffer          *input,
            GeglBuffer          *output,
            const GeglRectangle *result,
            const Babl          *format,
            gint                 patch_size,
            gfloat               threshold)
{
  static GeglClRunData *cl_data = NULL;
  const GeglRectangle  *extent = gegl_buffer_get_extent (input);
  const gfloat         *basis_data = patch_size == 8 ? &DCTbasis8x8[0][0]
                                                       : &DCTbasis16x16[0][0];
  gint                  margin = patch_size - 1;
  gboolean              success = TRUE;
  cl_mem                basis;
  cl_int                cl_err;

  if (! gegl_rectangle_contains (extent, result))
    return FALSE;

  g_mutex_lock (&cl_mutex);
  if (! cl_data)
    {
      const char *kernel_names[] = { "denoise_dct_forward_rows",
                                     "denoise_dct_columns",
                                     "denoise_dct_inverse_rows",
                                     "denoise_dct_average", NULL };
#if defined (__ARM_FEATURE_FMA) || defined (__FMA__)
      gchar *source = g_strconcat ("#define GEGL_DCT_FUSED 1\n",
                                   denoise_dct_cl_source, NULL);
      cl_data = gegl_cl_compile_and_build (source, kernel_names);
      g_free (source);
#else
      cl_data = gegl_cl_compile_and_build (denoise_dct_cl_source, kernel_names);
#endif
    }
  g_mutex_unlock (&cl_mutex);

  if (! cl_data)
    return FALSE;

  basis = gegl_clCreateBuffer (gegl_cl_get_context (),
                                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                patch_size * patch_size * sizeof (cl_float),
                                (gpointer) basis_data, &cl_err);
  if (cl_err != CL_SUCCESS)
    return FALSE;

  for (gint y = 0; y < result->height && success; y += CL_DCT_TILE_HEIGHT)
    {
      for (gint x = 0; x < result->width && success; x += CL_DCT_TILE_WIDTH)
        {
          GeglRectangle tile = { result->x + x, result->y + y,
                                 MIN (CL_DCT_TILE_WIDTH, result->width - x),
                                 MIN (CL_DCT_TILE_HEIGHT, result->height - y) };
          gboolean err = FALSE;
          GeglBufferClIterator *i =
            gegl_buffer_cl_iterator_new (output, &tile, format,
                                         GEGL_CL_BUFFER_WRITE);
          gint read = gegl_buffer_cl_iterator_add_2 (i, input, &tile, format,
                                                      GEGL_CL_BUFFER_READ,
                                                      margin, margin, margin, margin,
                                                      GEGL_ABYSS_NONE);

          while (gegl_buffer_cl_iterator_next (i, &err) && ! err)
            {
              GeglRectangle source_rect;

              gegl_rectangle_intersect (&source_rect, &i->roi[read], extent);
              g_mutex_lock (&cl_mutex);
              success = cl_denoise_tile (cl_data, i->tex[read], i->tex[0], basis,
                                         &i->roi[read], &source_rect, &i->roi[0],
                                         patch_size, threshold);
              g_mutex_unlock (&cl_mutex);

              if (! success)
                {
                  gegl_buffer_cl_iterator_stop (i);
                  break;
                }
            }

          success = success && ! err;
        }

      if (success)
        gegl_operation_progress (operation,
                                 (gdouble) MIN (y + CL_DCT_TILE_HEIGHT,
                                                result->height) / result->height,
                                 (gchar *) "");
    }

  gegl_clReleaseMemObject (basis);
  return success;
}

/* 1 dimensional Discret Cosine Transform of a signal of size 8x1. */

static void
dct_1d_8x8 (gfloat  *in,
            gfloat  *out,
            gboolean forward)
{
  gint  i, j;

  if (forward)
    {
      for (j = 0; j < 8; j++)
        {
          for (i = 0; i < 8; i++)
            {
              out[0] += in[i*3]   * DCTbasis8x8[j][i];
              out[1] += in[i*3+1] * DCTbasis8x8[j][i];
              out[2] += in[i*3+2] * DCTbasis8x8[j][i];
            }

          out += 3;
        }
    }
  else
    {
      for (j = 0; j < 8; j++)
        {
          for (i = 0; i < 8; i ++)
            {
              out[0] += in[i*3]   * DCTbasis8x8[i][j];
              out[1] += in[i*3+1] * DCTbasis8x8[i][j];
              out[2] += in[i*3+2] * DCTbasis8x8[i][j];
            }

          out += 3;
        }
    }
}

/* 1 dimensional Discret Cosine Transform of a signal of size 16x1. */

static void
dct_1d_16x16 (gfloat  *in,
              gfloat  *out,
              gboolean forward)
{
  gint  i, j;

  if (forward)
    {
      for (j = 0; j < 16; j++)
        {
          for (i = 0; i < 16; i++)
            {
              out[0] += in[i*3]   * DCTbasis16x16[j][i];
              out[1] += in[i*3+1] * DCTbasis16x16[j][i];
              out[2] += in[i*3+2] * DCTbasis16x16[j][i];
            }

          out += 3;
        }
    }
  else
    {
      for (j = 0; j < 16; j++)
        {
          for (i = 0; i < 16; i ++)
            {
              out[0] += in[i*3]   * DCTbasis16x16[i][j];
              out[1] += in[i*3+1] * DCTbasis16x16[i][j];
              out[2] += in[i*3+2] * DCTbasis16x16[i][j];
            }

          out += 3;
        }
    }
}

/* 2 dimensional DCT of a 8x8 (or 16x16) patch */

static void
dct_2d (gfloat   *patch,
        gint      patch_size,
        gboolean  forward)
{
  gfloat  *tmp1;
  gfloat  *tmp2;
  gint     x, y;

  tmp1 = g_new0 (gfloat, patch_size * patch_size * 3);
  tmp2 = g_new (gfloat, patch_size * patch_size * 3);

  /* transform row by row */

  if (patch_size == 8)
    {
      for (y = 0; y < 8; y++)
        {
          dct_1d_8x8 (patch + y * 8 * 3,
                      tmp1 + y * 8 * 3,
                      forward);
        }
    }
  else
    {
      for (y = 0; y < 16; y++)
        {
          dct_1d_16x16 (patch + y * 16 * 3,
                        tmp1 + y * 16 * 3,
                        forward);
        }
    }

  /* transform column by column (by transposing the matrix, transforming row by
   * row, and transposing again the matrix.)
   */

  for (y = 0; y < patch_size; y++)
    {
      for (x = 0; x < patch_size; x++)
        {
          tmp2[(y + x * patch_size) * 3]   = tmp1[(x + y * patch_size) * 3];
          tmp2[(y + x * patch_size) * 3+1] = tmp1[(x + y * patch_size) * 3+1];
          tmp2[(y + x * patch_size) * 3+2] = tmp1[(x + y * patch_size) * 3+2];
        }
    }

  memset (tmp1, 0.f, patch_size * patch_size * 3 * sizeof(gfloat));

  if (patch_size == 8)
    {
      for (y = 0; y < 8; y++)
        {
          dct_1d_8x8 (tmp2 + y * 8 * 3,
                      tmp1 + y * 8 * 3,
                      forward);
        }
    }
  else
    {
      for (y = 0; y < 16; y++)
        {
          dct_1d_16x16 (tmp2 + y * 16 * 3,
                        tmp1 + y * 16 * 3,
                        forward);
        }
    }

  for (y = 0; y < patch_size; y++)
    {
      for (x = 0; x < patch_size; x++)
        {
          patch[(y + x * patch_size) * 3]   = tmp1[(x + y * patch_size) * 3];
          patch[(y + x * patch_size) * 3+1] = tmp1[(x + y * patch_size) * 3+1];
          patch[(y + x * patch_size) * 3+2] = tmp1[(x + y * patch_size) * 3+2];
        }
    }

  g_free (tmp1);
  g_free (tmp2);
}

static void
threshold_patch_coefficients (gfloat  *patch,
                              gint     patch_n_pixels,
                              gfloat   threshold)
{
  gint     i;
  gfloat  *p = patch;

  for (i = 0; i < patch_n_pixels; i++)
    {
      p[0] = ABS (p[0]) < threshold ? 0.f : p[0];
      p[1] = ABS (p[1]) < threshold ? 0.f : p[1];
      p[2] = ABS (p[2]) < threshold ? 0.f : p[2];
      p += 3;
    }
}

static void
prepare (GeglOperation *operation)
{
  const Babl *space  = gegl_operation_get_source_space (operation, "input");
  const Babl *format = babl_format_with_space ("R'G'B'A float", space);

  gegl_operation_set_format (operation, "input", format);
  gegl_operation_set_format (operation, "output", format);
}

static GeglRectangle
get_cached_region (GeglOperation       *operation,
                   const GeglRectangle *roi)
{
  const GeglRectangle *in_rect =
      gegl_operation_source_get_bounding_box (operation, "input");

  if (! in_rect || gegl_rectangle_is_infinite_plane (in_rect))
    return *roi;

  return *in_rect;
}

static GeglRectangle
get_required_for_output (GeglOperation       *operation,
                         const gchar         *input_pad,
                         const GeglRectangle *roi)
{
  return get_cached_region (operation, roi);
}

static gboolean
process (GeglOperation       *operation,
         GeglBuffer          *input,
         GeglBuffer          *output,
         const GeglRectangle *result,
         gint                 level)
{
  GeglProperties *o  = GEGL_PROPERTIES (operation);
  const Babl *space  = gegl_operation_get_source_space (operation, "input");
  const Babl *rgb_f  = babl_format_with_space ("R'G'B' float", space);
  const Babl *rgba_f = babl_format_with_space ("R'G'B'A float", space);
  const GeglRectangle *extent = gegl_buffer_get_extent (input);

  GeglBuffer  *sum;
  gint    width, height;
  gint    patch_size;
  gint    patch_len;
  gfloat  threshold;
  gdouble progress;
  gint   *patch_n_x;
  gint   *patch_n_y;
  gint    i;

  width  = extent->width;
  height = extent->height;
  patch_size = o->patch_size == GEGL_DENOISE_DCT_8X8 ? 8 : 16;
  patch_len = patch_size * patch_size;
  threshold = 3.f * (gfloat) o->sigma / 255.;

  gegl_operation_progress (operation, 0.0, (gchar *) "");

  if (level == 0 && gegl_operation_use_opencl (operation) &&
      cl_denoise (operation, input, output, result, rgba_f,
                  patch_size, threshold))
    return TRUE;

  sum = gegl_buffer_new (extent, rgb_f);

  patch_n_x = g_new (gint, width);
  patch_n_y = g_new (gint, height);

  GeglBufferIterator  *iter;
  gint                 x_offset;

  for (x_offset = 0; x_offset < patch_size; x_offset++)
    {
      /* Input buffer is split into vertical non-overlapping regions.
       * Regions are distributed among threads.
       */

      gint n_regions = (width - x_offset) / patch_size;

      gegl_parallel_distribute_range (
        n_regions, gegl_operation_get_pixels_per_thread (operation)
           / (height * patch_size),
        [=] (gint region0, gint n)
        {
          gint     x, y, j;
          gfloat  *in_buf;
          gfloat  *sum_buf;
          gfloat  *patch_buf;

          in_buf    = g_new (gfloat, patch_size * height * 3);
          sum_buf   = g_new (gfloat, patch_size * height * 3);
          patch_buf = g_new (gfloat, patch_len * 3);

          for (x = region0; x < region0 + n; x++)
            {
              /* Threads manage one region at a time:
               * for each overlapping patch from top to bottom:
               *    1. extract patch
               *    2. dc transform, thresholding, inverse transform
               *    3. sum the patch result into the sum memory buffer
               * set the gegl buffer with the sum memory buffer
               */

              GeglRectangle  roi = {extent->x + x * patch_size + x_offset,
                                    extent->y,
                                    patch_size,
                                    height};
              gint patch_offset  = 0;

              gegl_buffer_get (input, &roi, 1.0, rgb_f, in_buf,
                               GEGL_AUTO_ROWSTRIDE, GEGL_ABYSS_NONE);

              gegl_buffer_get (sum, &roi, 1.0, rgb_f, sum_buf,
                               GEGL_AUTO_ROWSTRIDE, GEGL_ABYSS_NONE);

              for (y = 0; y < height - patch_size + 1; y++)
                {
                  gfloat  *sum_p   = sum_buf + patch_offset;
                  gfloat  *patch_p = patch_buf;

                  memcpy (patch_buf, in_buf + patch_offset,
                          patch_len * 3 * sizeof(gfloat));
                  dct_2d (patch_buf, patch_size, TRUE);
                  threshold_patch_coefficients (patch_buf, patch_len, threshold);
                  dct_2d (patch_buf, patch_size, FALSE);

                  for (j = 0; j < patch_len * 3; j++)
                    {
                      *sum_p++ += *patch_p++;
                    }

                  patch_offset += patch_size * 3;
                }

              gegl_buffer_set (sum, &roi, 0, rgb_f, sum_buf,
                               GEGL_AUTO_ROWSTRIDE);
            }

          g_free (in_buf);
          g_free (sum_buf);
          g_free (patch_buf);
        });

      progress = (gdouble) (x_offset + 1) / (gdouble) patch_size;
      gegl_operation_progress (operation, progress, (gchar *) "");
    }

  /* Finally, average the accumulated values of the sum buffer, given the
   *  appropriate number of patches a pixel belongs to.
   *  Put the result in the ouput buffer with the original alpha value.
   */

  for (i = 0; i < patch_size; i++)
    {
      patch_n_x[i] = patch_n_x[width -1 - i] = i+1;
      patch_n_y[i] = patch_n_y[height -1 -i] = i+1;
    }

  for (i = patch_size; i <= width - patch_size; i++)
    patch_n_x[i] = patch_size;

  for (i = patch_size; i <= height - patch_size; i++)
    patch_n_y[i] = patch_size;

  iter = gegl_buffer_iterator_new (input, NULL, 0, rgba_f,
                                   GEGL_ACCESS_READ, GEGL_ABYSS_NONE, 3);

  gegl_buffer_iterator_add (iter, sum, NULL, 0, rgb_f,
                            GEGL_ACCESS_READ, GEGL_ABYSS_NONE);

  gegl_buffer_iterator_add (iter, output, NULL, 0, rgba_f,
                            GEGL_ACCESS_WRITE, GEGL_ABYSS_NONE);

  while (gegl_buffer_iterator_next (iter))
    {
      gfloat  *in   = (gfloat *) iter->items[0].data;
      gfloat  *sum  = (gfloat *) iter->items[1].data;
      gfloat  *out  = (gfloat *) iter->items[2].data;
      GeglRectangle *roi = &iter->items[0].roi;
      gint     x, y;

      for (y = roi->y; y < roi->y + roi->height; y++)
        {
          for (x = roi->x; x < roi->x + roi->width; x++)
            {
              gint   n_patches = patch_n_x[x - extent->x] *
                                 patch_n_y[y - extent->y];
              gfloat n = 1.f / (gfloat) n_patches;

              out[0] = sum[0] * n;
              out[1] = sum[1] * n;
              out[2] = sum[2] * n;
              out[3] = in[3];

              in  += 4;
              sum += 3;
              out += 4;
            }
        }
    }

  gegl_operation_progress (operation, 1.0, (gchar *) "");

  g_object_unref (sum);
  g_free (patch_n_x);
  g_free (patch_n_y);

  return TRUE;
}

static gboolean
operation_process (GeglOperation        *operation,
                   GeglOperationContext *context,
                   const gchar          *output_prop,
                   const GeglRectangle  *result,
                   gint                  level)
{
  GeglOperationClass  *operation_class;
  gint                 min_size;
  GeglProperties      *o = GEGL_PROPERTIES (operation);

  min_size = o->patch_size == GEGL_DENOISE_DCT_8X8 ? 16 : 32;

  const GeglRectangle *in_rect =
    gegl_operation_source_get_bounding_box (operation, "input");

  if (in_rect && (gegl_rectangle_is_infinite_plane (in_rect) ||
                  in_rect->width < min_size ||
                  in_rect->height < min_size))
    {
      gpointer in = gegl_operation_context_get_object (context, "input");
      gegl_operation_context_take_object (context, "output",
                                          G_OBJECT(g_object_ref (in)));
      return TRUE;
    }

  operation_class = GEGL_OPERATION_CLASS (gegl_op_parent_class);

  return operation_class->process (operation, context, output_prop, result,
                                   gegl_operation_context_get_level (context));
}

static void
gegl_op_class_init (GeglOpClass *klass)
{
  GeglOperationClass       *operation_class;
  GeglOperationFilterClass *filter_class;

  operation_class = GEGL_OPERATION_CLASS (klass);
  filter_class    = GEGL_OPERATION_FILTER_CLASS (klass);

  operation_class->threaded                = FALSE;
  operation_class->opencl_support          = TRUE;
  operation_class->prepare                 = prepare;
  operation_class->process                 = operation_process;
  operation_class->get_cached_region       = get_cached_region;
  operation_class->get_required_for_output = get_required_for_output;
  filter_class->process                    = process;

  gegl_operation_class_set_keys (operation_class,
    "name",        "gegl:denoise-dct",
    "title",       _("Denoise DCT"),
    "categories",  "enhance:noise-reduction",
    "description", _("Denoising algorithm using a per-patch DCT thresholding"),
    NULL);
}

#endif
