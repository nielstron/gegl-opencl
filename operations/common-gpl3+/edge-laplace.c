/* This file is an image processing operation for GEGL
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 */

/*
 * Copyright 1997 Thorsten Schnier <thorsten@arch.usyd.edu.au>
 * Copyright 2011 Victor Oliveira <victormatheus@gmail.com>
 */

#include "config.h"
#include <glib/gi18n-lib.h>


#ifdef GEGL_PROPERTIES

#else

#define GEGL_OP_AREA_FILTER
#define GEGL_OP_NAME     edge_laplace
#define GEGL_OP_C_SOURCE edge-laplace.c

#include "gegl-op.h"

#define LAPLACE_RADIUS 2
#define CHUNK_SIZE     1024
#define SQR(x)         ((x)*(x))

static void
edge_laplace (GeglBuffer          *src,
              const GeglRectangle *src_rect,
              GeglBuffer          *dst,
              const GeglRectangle *dst_rect,
              gfloat              *src_buf,
              gfloat              *temp_buf,
              gfloat              *dst_buf,
              const Babl          *format);

static void
prepare (GeglOperation *operation)
{
  GeglOperationAreaFilter *area = GEGL_OPERATION_AREA_FILTER (operation);
  const Babl *space = gegl_operation_get_source_space (operation, "input");

  area->left = area->right = area->top = area->bottom = LAPLACE_RADIUS;

  gegl_operation_set_format (operation, "input", babl_format_with_space ("R'G'B'A float", space));
  gegl_operation_set_format (operation, "output", babl_format_with_space ("R'G'B'A float", space));
}

static gboolean
cl_process (GeglOperation       *operation,
            GeglBuffer          *input,
            GeglBuffer          *output,
            const GeglRectangle *result);

static gboolean
process (GeglOperation       *operation,
         GeglBuffer          *input,
         GeglBuffer          *output,
         const GeglRectangle *result,
         gint                 level)
{
  gint    i, j;
  gfloat *buf1, *buf2, *buf3;
  const Babl *format;

  if (gegl_operation_use_opencl (operation))
    if (cl_process (operation, input, output, result))
      return TRUE;
  format = gegl_operation_get_format (operation, "output");

  buf1 = g_new (gfloat, SQR (CHUNK_SIZE + LAPLACE_RADIUS * 2) * 4);
  buf2 = g_new (gfloat, SQR (CHUNK_SIZE + LAPLACE_RADIUS * 2) * 4);
  buf3 = g_new (gfloat, SQR (CHUNK_SIZE) * 4);

  for (j = 0; (j-1) * CHUNK_SIZE < result->height; j++)
    for (i = 0; (i-1) * CHUNK_SIZE < result->width; i++)
      {
        GeglRectangle chunked_result;
        GeglRectangle compute;

        chunked_result = *GEGL_RECTANGLE (result->x + i * CHUNK_SIZE,
                                          result->y + j * CHUNK_SIZE,
                                          CHUNK_SIZE, CHUNK_SIZE);

        gegl_rectangle_intersect (&chunked_result, &chunked_result, result);

        if (chunked_result.width < 1  || chunked_result.height < 1)
          continue;

        compute = gegl_operation_get_required_for_output (operation,
                                                          "input",
                                                          &chunked_result);

        edge_laplace (input, &compute, output, &chunked_result,
                      buf1, buf2, buf3, format);
      }

  g_free (buf1);
  g_free (buf2);
  g_free (buf3);

  return  TRUE;
}

static void
minmax  (gfloat  x1,
         gfloat  x2,
         gfloat  x3,
         gfloat  x4,
         gfloat  x5,
         gfloat *min_result,
         gfloat *max_result)
{
  gfloat min1, min2, max1, max2;

  if (x1 > x2)
    {
      max1 = x1;
      min1 = x2;
    }
  else
    {
      max1 = x2;
      min1 = x1;
    }

  if (x3 > x4)
    {
      max2 = x3;
      min2 = x4;
    }
  else
    {
      max2 = x4;
      min2 = x3;
    }

  if (min1 < min2)
    *min_result = fminf (min1, x5);
  else
    *min_result = fminf (min2, x5);

  if (max1 > max2)
    *max_result = fmaxf (max1, x5);
  else
    *max_result = fmaxf (max2, x5);
}


static void
edge_laplace (GeglBuffer          *src,
              const GeglRectangle *src_rect,
              GeglBuffer          *dst,
              const GeglRectangle *dst_rect,
              gfloat              *src_buf,
              gfloat              *temp_buf,
              gfloat              *dst_buf,
              const Babl          *format)
{

  gint    x, y;
  gint    offset;
  gint    src_width = src_rect->width;

  gegl_buffer_get (src, src_rect, 1.0,
                   format, src_buf, GEGL_AUTO_ROWSTRIDE,
                   GEGL_ABYSS_CLAMP);

  for (y = 0; y < dst_rect->height + LAPLACE_RADIUS; y++)
    for (x = 0; x < dst_rect->width + LAPLACE_RADIUS; x++)
      {
        gfloat *src_pix;
        gfloat  gradient[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        gint    c;
        gfloat  minval, maxval;
        gint    i = x + LAPLACE_RADIUS - 1;
        gint    j = y + LAPLACE_RADIUS - 1;

        offset = i + j * src_width;
        src_pix = src_buf + offset * 4;

        for (c = 0; c < 3; c++)
          {
            gfloat s1 = src_pix[c - src_width * 4];
            gfloat s2 = src_pix[c + src_width * 4];
            gfloat s3 = src_pix[c - 4];
            gfloat s4 = src_pix[c + 4];
            gfloat s  = src_pix[c];
            gfloat temp_value;

            /* four-neighbourhood */
            minmax (s1, s2, s3, s4, s,
                    &minval, &maxval);

            gradient[c] = 0.5f * fmaxf ((maxval - s), (s - minval));

            /* nine-neighbourhood */
            temp_value = (src_pix[c - 4 - src_width * 4] +
                          s1 +
                          src_pix[c + 4 - src_width * 4] +

                          s3 - 8.0f * s + s4 +

                          src_pix[c - 4 + src_width * 4] +
                          s2 +
                          src_pix[c + 4 + src_width * 4]);


            if (temp_value < GEGL_FLOAT_EPSILON)
              gradient[c] *= -1.0f;
          }

        /* alpha */
        gradient[3] = src_pix[3];

        for (c = 0; c < 4; c++)
          temp_buf[offset * 4 + c] = gradient[c];
      }

  /* 1-pixel edges */
  offset = 0;

  for (y = 0; y < dst_rect->height; y++)
    for (x = 0; x < dst_rect->width; x++)
      {
        gfloat  value[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        gint    c;
        gint    i = x + LAPLACE_RADIUS;
        gint    j = y + LAPLACE_RADIUS;
        gfloat *src_pix = temp_buf + (i + j * src_width) * 4;

        for (c = 0; c < 3; c++)
          {
            gfloat current = src_pix[c];

            current = ((current > 0.0f) &&
                       (src_pix[c - 4 - src_width * 4] < 0.0f ||
                        src_pix[c + 4 - src_width * 4] < 0.0f ||
                        src_pix[c     - src_width * 4] < 0.0f ||
                        src_pix[c - 4 + src_width * 4] < 0.0f ||
                        src_pix[c + 4 + src_width * 4] < 0.0f ||
                        src_pix[c     + src_width * 4] < 0.0f ||
                        src_pix[c - 4                ] < 0.0f ||
                        src_pix[c + 4                ] < 0.0f)) ?
              current : 0.0f;

            value[c] = current;
          }

        /* alpha */
        value[3] = src_pix[3];

        for (c = 0; c < 4; c++)
          dst_buf[offset * 4 + c] = value[c];

        offset++;
      }

  gegl_buffer_set (dst, dst_rect, 0, format, dst_buf,
                   GEGL_AUTO_ROWSTRIDE);
}

#include "opencl/gegl-cl.h"
#include "gegl-buffer-cl-iterator.h"

#include "opencl/edge-laplace.cl.h"

static gboolean
cl_process (GeglOperation       *operation,
            GeglBuffer          *input,
            GeglBuffer          *output,
            const GeglRectangle *result)
{
  static GeglClRunData *cl_data;
  const Babl *in_format  = gegl_operation_get_format (operation, "input");
  const Babl *out_format = gegl_operation_get_format (operation, "output");
  GeglOperationAreaFilter *op_area = GEGL_OPERATION_AREA_FILTER (operation);
  GeglBufferClIterator *iter;
  size_t local_ws[] = { 16, 8 };
  size_t tile_bytes;
  gboolean err;
  cl_int cl_err;
  gint read;

  if (!cl_data)
    {
      const char *kernels[] = { "edge_laplace", NULL };
      cl_data = gegl_cl_compile_and_build (edge_laplace_cl_source, kernels);
    }
  if (!cl_data || cl_data->work_group_size[0] == 0)
    return FALSE;

  while (local_ws[0] * local_ws[1] > cl_data->work_group_size[0])
    {
      if (local_ws[0] > local_ws[1])
        local_ws[0] /= 2;
      else
        local_ws[1] /= 2;
    }
  tile_bytes = (local_ws[0] + 2) * (local_ws[1] + 2) * sizeof (cl_float4);
  if (tile_bytes > gegl_cl_get_local_mem_size ())
    return FALSE;

  iter = gegl_buffer_cl_iterator_new (output, result, out_format, GEGL_CL_BUFFER_WRITE);
  read = gegl_buffer_cl_iterator_add_2 (iter, input, result, in_format,
                                        GEGL_CL_BUFFER_READ,
                                        op_area->left, op_area->right,
                                        op_area->top, op_area->bottom,
                                        GEGL_ABYSS_CLAMP);

  while (gegl_buffer_cl_iterator_next (iter, &err))
    {
      const cl_int width = iter->roi[0].width;
      const cl_int height = iter->roi[0].height;
      const size_t global_ws[] = {
        ((size_t) width  + local_ws[0] - 1) / local_ws[0] * local_ws[0],
        ((size_t) height + local_ws[1] - 1) / local_ws[1] * local_ws[1]
      };

      cl_err = gegl_cl_set_kernel_args (cl_data->kernel[0],
                                        sizeof (cl_mem), &iter->tex[read],
                                        sizeof (cl_mem), &iter->tex[0],
                                        sizeof (cl_int), &width,
                                        sizeof (cl_int), &height,
                                        tile_bytes, NULL,
                                        NULL);
      CL_CHECK;

      cl_err = gegl_clEnqueueNDRangeKernel (gegl_cl_get_command_queue (),
                                            cl_data->kernel[0], 2, NULL,
                                            global_ws, local_ws, 0, NULL, NULL);
      CL_CHECK;
    }

  return !err;

error:
  gegl_buffer_cl_iterator_stop (iter);
  return FALSE;
}

static void
gegl_op_class_init (GeglOpClass *klass)
{
  GeglOperationClass       *operation_class;
  GeglOperationFilterClass *filter_class;

  operation_class = GEGL_OPERATION_CLASS (klass);
  filter_class    = GEGL_OPERATION_FILTER_CLASS (klass);

  filter_class->process    = process;
  operation_class->prepare = prepare;

  operation_class->opencl_support = TRUE;

  gegl_operation_class_set_keys (operation_class,
    "name",           "gegl:edge-laplace",
    "title",          _("Laplacian Edge Detection"),
    "categories",     "edge-detect",
    "license",        "GPL3+",
    "reference-hash", "c4ecceb172485fb4550e4b31a43eb2b1",
    "description",    _("High-resolution edge detection"),
    NULL);
}

#endif
