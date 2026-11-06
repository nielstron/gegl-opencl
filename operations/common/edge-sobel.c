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
 */

/*
 * Copyright 2011 Victor Oliveira <victormatheus@gmail.com>
 */

#include "config.h"
#include <glib/gi18n-lib.h>

#ifdef GEGL_PROPERTIES

property_boolean (horizontal,  _("Horizontal"), TRUE)

property_boolean (vertical,  _("Vertical"), TRUE)

property_boolean (keep_sign,  _("Keep sign"), TRUE)
     description (_("Keep negative values in result; when off, the absolute value of the result is used instead."))

#else

#define GEGL_OP_AREA_FILTER
#define GEGL_OP_NAME     edge_sobel
#define GEGL_OP_C_SOURCE edge-sobel.c

#include "gegl-op.h"
#include <stdio.h>
#include <string.h>

#define SOBEL_RADIUS 1

static void
edge_sobel (GeglBuffer          *src,
            const GeglRectangle *src_rect,
            GeglBuffer          *dst,
            const GeglRectangle *dst_rect,
            gboolean            horizontal,
            gboolean            vertical,
            gboolean            keep_sign,
            gboolean            has_alpha,
            const Babl         *format);



static void prepare (GeglOperation *operation)
{
  const Babl *space = gegl_operation_get_source_space (operation, "input");
  GeglOperationAreaFilter *area = GEGL_OPERATION_AREA_FILTER (operation);

  const Babl *format = babl_format_with_space ("RGBA float", space);

  area->left = area->right = area->top = area->bottom = SOBEL_RADIUS;

  gegl_operation_set_format (operation, "input", format);
  gegl_operation_set_format (operation, "output", format);
}

#include "opencl/gegl-cl.h"
#include "gegl-buffer-cl-iterator.h"

#include "opencl/edge-sobel.cl.h"

static GeglClRunData *cl_data = NULL;

/* Tables for converting between "R'G'B'A u8" and "RGBA float" in the
 * kernel, see edge-sobel.cl.
 */
#define LUT_SIZE 1024

/* Fills lut[1..255] with the smallest non-negative float that babl
 * converts to k or more, for the color channels or for the alpha.
 */
static void
make_quantization_lut (const Babl *fish,
                       gboolean    alpha,
                       gfloat     *lut)
{
  gfloat  src[255 * 4];
  guint8  dst[255 * 4];
  guint32 lo[256];
  guint32 hi[256];
  gint    k, c;
  gint    channel = alpha ? 3 : 0;

  /* the bit patterns of non-negative floats are in the same order as the
   * floats; invariant: babl (lo[k]) < k <= babl (hi[k])
   */
  for (k = 1; k < 256; k++)
    {
      gfloat zero = 0.0f;
      gfloat two  = 2.0f;

      memcpy (&lo[k], &zero, sizeof (gfloat));
      memcpy (&hi[k], &two, sizeof (gfloat));
    }

  for (gint iteration = 0; iteration < 32; iteration++)
    {
      for (k = 1; k < 256; k++)
        {
          guint32 mid = lo[k] + (hi[k] - lo[k]) / 2;
          gfloat  value;

          memcpy (&value, &mid, sizeof (gfloat));

          for (c = 0; c < 3; c++)
            src[(k - 1) * 4 + c] = alpha ? 0.0f : value;
          src[(k - 1) * 4 + 3] = alpha ? value : 1.0f;
        }

      babl_process (fish, src, dst, 255);

      for (k = 1; k < 256; k++)
        {
          guint32 mid = lo[k] + (hi[k] - lo[k]) / 2;

          if (dst[(k - 1) * 4 + channel] >= k)
            hi[k] = mid;
          else
            lo[k] = mid;
        }
    }

  lut[0] = -G_MAXFLOAT; /* never read */
  for (k = 1; k < 256; k++)
    memcpy (&lut[k], &hi[k], sizeof (gfloat));
}

static cl_mem
get_lut (const Babl *u8_format,
         const Babl *float_format)
{
  static cl_mem lut_mem = NULL;
  static GMutex mutex;
  gfloat        lut[LUT_SIZE];
  guint8        src[256 * 4];
  gfloat        dst[256 * 4];
  const Babl   *fish;
  cl_int        cl_err = 0;
  gint          k;

  g_mutex_lock (&mutex);

  if (lut_mem)
    {
      g_mutex_unlock (&mutex);
      return lut_mem;
    }

  fish = babl_fish (u8_format, float_format);

  for (k = 0; k < 256; k++)
    {
      src[4 * k + 0] = src[4 * k + 1] = src[4 * k + 2] = k;
      src[4 * k + 3] = 255;
    }
  babl_process (fish, src, dst, 256);
  for (k = 0; k < 256; k++)
    lut[k] = dst[4 * k];

  for (k = 0; k < 256; k++)
    {
      src[4 * k + 0] = src[4 * k + 1] = src[4 * k + 2] = 0;
      src[4 * k + 3] = k;
    }
  babl_process (fish, src, dst, 256);
  for (k = 0; k < 256; k++)
    lut[256 + k] = dst[4 * k + 3];

  fish = babl_fish (float_format, u8_format);

  make_quantization_lut (fish, FALSE, lut + 512);
  make_quantization_lut (fish, TRUE,  lut + 768);

  lut_mem = gegl_clCreateBuffer (gegl_cl_get_context (),
                                 CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 sizeof (lut), lut, &cl_err);
  if (cl_err != CL_SUCCESS)
    lut_mem = NULL;

  g_mutex_unlock (&mutex);

  return lut_mem;
}

static gboolean
cl_edge_sobel (cl_mem              in_tex,
               cl_mem              out_tex,
               cl_mem              lut,
               const GeglRectangle *roi,
               gboolean            horizontal,
               gboolean            vertical,
               gboolean            keep_sign,
               gboolean            has_alpha,
               gboolean            at_left_edge,
               gboolean            at_top_edge,
               gboolean            in_u8,
               gboolean            out_u8)
{
  static size_t local_size[2] = {0, 0};
  size_t gbl_size[2];
  cl_int n_width       = roi->width;
  cl_int n_height      = roi->height;
  cl_int n_horizontal  = horizontal;
  cl_int n_vertical    = vertical;
  cl_int n_keep_sign   = keep_sign;
  cl_int n_has_alpha   = has_alpha;
  cl_int min_x         = at_left_edge ? 1 : 0;
  cl_int min_y         = at_top_edge  ? 1 : 0;
  cl_int n_in_u8       = in_u8;
  cl_int n_out_u8      = out_u8;
  cl_int cl_err = 0;

  if (!cl_data)
    {
      const char *kernel_name[] = {"kernel_edgesobel", NULL};
      cl_data = gegl_cl_compile_and_build (edge_sobel_cl_source, kernel_name);
    }
  if (!cl_data) return TRUE;

  if (! local_size[0])
    {
      size_t work_group_size;

      cl_err = gegl_clGetKernelWorkGroupInfo (cl_data->kernel[0],
                                              gegl_cl_get_device (),
                                              CL_KERNEL_WORK_GROUP_SIZE,
                                              sizeof (size_t),
                                              &work_group_size,
                                              NULL);
      CL_CHECK;

      /* 16x16 work-items, or fewer where the device allows less */
      local_size[0] = 16;
      local_size[1] = 16;
      while (local_size[0] * local_size[1] > work_group_size &&
             local_size[1] > 1)
        {
          if (local_size[1] >= local_size[0])
            local_size[1] /= 2;
          else
            local_size[0] /= 2;
        }
    }

  gbl_size[0] = (roi->width  + local_size[0] - 1) / local_size[0] * local_size[0];
  gbl_size[1] = (roi->height + local_size[1] - 1) / local_size[1] * local_size[1];

  cl_err = gegl_cl_set_kernel_args (cl_data->kernel[0],
                                    sizeof (cl_mem), &in_tex,
                                    sizeof (cl_mem), &out_tex,
                                    sizeof (cl_mem), &lut,
                                    (local_size[0] + 2) * (local_size[1] + 2) *
                                    sizeof (cl_float4), NULL,
                                    512 * sizeof (cl_float), NULL,
                                    sizeof (cl_int), &n_width,
                                    sizeof (cl_int), &n_height,
                                    sizeof (cl_int), &n_horizontal,
                                    sizeof (cl_int), &n_vertical,
                                    sizeof (cl_int), &n_keep_sign,
                                    sizeof (cl_int), &n_has_alpha,
                                    sizeof (cl_int), &min_x,
                                    sizeof (cl_int), &min_y,
                                    sizeof (cl_int), &n_in_u8,
                                    sizeof (cl_int), &n_out_u8,
                                    NULL);
  CL_CHECK;

  cl_err = gegl_clEnqueueNDRangeKernel(gegl_cl_get_command_queue(),
                                       cl_data->kernel[0], 2,
                                       NULL, gbl_size, local_size,
                                       0, NULL, NULL);
  CL_CHECK;

  return FALSE;

error:
  return TRUE;
}

static gboolean
cl_process (GeglOperation       *operation,
            GeglBuffer          *input,
            GeglBuffer          *output,
            const GeglRectangle *result,
            gboolean             has_alpha)
{
  const Babl *float_format = babl_format ("RGBA float");
  const Babl *u8_format    = babl_format ("R'G'B'A u8");
  const Babl *in_format    = float_format;
  const Babl *out_format   = float_format;
  cl_mem      lut          = NULL;
  gint err;

  GeglOperationAreaFilter *op_area = GEGL_OPERATION_AREA_FILTER (operation);
  GeglProperties *o = GEGL_PROPERTIES (operation);

  /* Read and write 8-bit buffers as they are and convert in the kernel,
   * instead of converting them to and from float in separate passes,
   * which also moves four times the data.  This needs the operation to
   * work in "RGBA float" of the same space as the 8-bit format.
   */
  if (gegl_operation_get_format (operation, "output") == float_format &&
      (gegl_buffer_get_format (input)  == u8_format ||
       gegl_buffer_get_format (output) == u8_format))
    {
      lut = get_lut (u8_format, float_format);

      if (lut)
        {
          if (gegl_buffer_get_format (input) == u8_format)
            in_format = u8_format;
          if (gegl_buffer_get_format (output) == u8_format)
            out_format = u8_format;
        }
    }

  /* The CPU code centres output pixel (x, y) on input pixel (x - 1, y - 1),
   * see the kernel; request the matching asymmetric halo.
   */
  GeglBufferClIterator *i = gegl_buffer_cl_iterator_new (output,
                                                         result,
                                                         out_format,
                                                         GEGL_CL_BUFFER_WRITE);

  gint read = gegl_buffer_cl_iterator_add_2 (i,
                                             input,
                                             result,
                                             in_format,
                                             GEGL_CL_BUFFER_READ,
                                             op_area->left + op_area->right,
                                             0,
                                             op_area->top + op_area->bottom,
                                             0,
                                             GEGL_ABYSS_NONE);

  while (gegl_buffer_cl_iterator_next (i, &err))
    {
      /* the kernel does not read the table for float formats, but it
       * still needs a valid buffer
       */
      err = cl_edge_sobel(i->tex[read],
                          i->tex[0],
                          lut ? lut : i->tex[read],
                          &i->roi[0],
                          o->horizontal,
                          o->vertical,
                          o->keep_sign,
                          has_alpha,
                          i->roi[0].x == result->x,
                          i->roi[0].y == result->y,
                          in_format == u8_format,
                          out_format == u8_format);

      if (err)
        {
          gegl_buffer_cl_iterator_stop (i);
          return FALSE;
        }
    }

  /* the iterator also stops on errors */
  return ! err;
}

static gboolean
process (GeglOperation       *operation,
         GeglBuffer          *input,
         GeglBuffer          *output,
         const GeglRectangle *result,
         gint                 level)
{
  GeglProperties   *o = GEGL_PROPERTIES (operation);
  GeglRectangle compute;
  gboolean has_alpha;

  compute = gegl_operation_get_required_for_output (operation, "input", result);
  has_alpha = babl_format_has_alpha (gegl_operation_get_format (operation, "output"));

  if (gegl_operation_use_opencl (operation))
    if (cl_process (operation, input, output, result, has_alpha))
      return TRUE;

  edge_sobel (input, &compute, output, result,
              o->horizontal, o->vertical, o->keep_sign, has_alpha,
              babl_format_with_space ("RGBA float",
              gegl_operation_get_format (operation, "output")));
  return TRUE;
}

inline static gfloat
magnitude(gfloat a, gfloat b)
{
  return sqrtf(a*a+b*b);
}

static void
edge_sobel (GeglBuffer          *src,
            const GeglRectangle *src_rect,
            GeglBuffer          *dst,
            const GeglRectangle *dst_rect,
            gboolean            horizontal,
            gboolean            vertical,
            gboolean            keep_sign,
            gboolean            has_alpha,
            const Babl         *format)
{
  gint x,y;
  gint offset;
  gfloat *src_buf;
  gfloat *after_src_buf;
  gfloat *dst_buf;

  src_buf = g_new0 (gfloat, src_rect->width * src_rect->height * 4);
  dst_buf = g_new0 (gfloat, dst_rect->width * dst_rect->height * 4);

  gegl_buffer_get (src, src_rect, 1.0, format,
                   src_buf, GEGL_AUTO_ROWSTRIDE, GEGL_ABYSS_NONE);

  after_src_buf = src_buf + src_rect->width * src_rect->height * 4;

  /* Apply the Sobel operator. Technically, the following is not Sobel
     as it does not use pixel intensities, but works on the individual
     RGB channels. But it is similar to the classic GIMP and PhotoShop
     filter.

     See paper "History and Definition of the Sobel Operator" by Irwin
     Sobel that seems to be the only free and authentic description.
  */
  offset = 0;
  for (y = 0; y < dst_rect->height; y++)
    for (x = 0; x < dst_rect->width; x++)
      {
        gfloat hor_grad[3] = {0.0f, 0.0f, 0.0f};
        gfloat ver_grad[3] = {0.0f, 0.0f, 0.0f};
        gfloat gradient[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        gfloat *row_start, *row_next;
        gfloat *tl_px, *t_px, *tr_px;
        gfloat *l_px, *center_px, *r_px;
        gfloat *bl_px, *b_px, *br_px;
        gint c;

        row_start = src_buf + y * src_rect->width * 4;
        row_next = row_start + src_rect->width * 4;

        /* This pixel */
        center_px = row_start + x * 4;
        /* Top pixel */
        t_px = center_px - src_rect->width * 4;
        /* Top-left pixel */
        tl_px = t_px - 4;
        /* Top-right pixel */
        tr_px = t_px + 4;
        /* Left pixel */
        l_px = center_px - 4;
        /* Right pixel */
        r_px = center_px + 4;
        /* Bottom pixel */
        b_px = center_px + src_rect->width * 4;
        /* Bottom-left pixel */
        bl_px = b_px - 4;
        /* Bottom-right pixel */
        br_px = b_px + 4;

        /* If pixels around the center pixel are out of source rect
           bounds, clamp them into the rect. */
        if (t_px < src_buf)
          {
            tl_px += src_rect->width * 4;
            t_px += src_rect->width * 4;
            tr_px += src_rect->width * 4;
          }
        else if (b_px >= after_src_buf)
          {
            bl_px -= src_rect->width * 4;
            b_px -= src_rect->width * 4;
            br_px -= src_rect->width * 4;
          }

        if (l_px < row_start)
          {
            tl_px += 4;
            l_px += 4;
            bl_px += 4;
          }
        else if (r_px >= row_next)
          {
            tr_px -= 4;
            r_px -= 4;
            br_px -= 4;
          }

        if (horizontal)
          {
            /*
             * Horizontal kernel:
             *
             *      [-1  0  +1]
             * Gx = [-2  0  +2] * P
             *      [-1  0  -1]
             */
            for (c = 0; c < 3; c++)
              {
                hor_grad[c] += (-1.0f * tl_px[c]) + (1.0f * tr_px[c]);
                hor_grad[c] += (-2.0f * l_px[c]) + (2.0f * r_px[c]);
                hor_grad[c] += (-1.0f * bl_px[c]) + (1.0f * br_px[c]);
              }
          }

        if (vertical)
          {
            /*
             * Vertical kernel:
             *
             *      [+1  +2  +1]
             * Gy = [ 0   0   0] * P
             *      [-1  -2  -1]
             */
            for (c = 0; c < 3; c++)
              {
                ver_grad[c] += (1.0f * tl_px[c]) + (2.0f * t_px[c]) + (1.0f * tr_px[c]);
                ver_grad[c] += (-1.0f * bl_px[c]) + (-2.0f * b_px[c]) + (-1.0f * br_px[c]);
              }
          }

        if (horizontal && vertical)
          {
             /* sqrt(32.0) = 5.656854 */
            for (c = 0; c < 3; c++)
              gradient[c] = magnitude (hor_grad[c], ver_grad[c]) / 5.656854f;
          }
        else
          {
            if (keep_sign)
              {
                for (c = 0; c < 3; c++)
                  gradient[c] = 0.5f + (hor_grad[c] + ver_grad[c]) / 8.0f;
              }
            else
              {
                for (c = 0; c < 3; c++)
                  gradient[c] = fabsf (hor_grad[c] + ver_grad[c]) / 4.0f;
              }
          }

        if (has_alpha)
          gradient[3] = center_px[3];
        else
          gradient[3] = 1.0f;

        for (c = 0; c < 4; c++)
          dst_buf[offset * 4 + c] = gradient[c];

        offset++;
      }

  gegl_buffer_set (dst, dst_rect, 0, format, dst_buf,
                   GEGL_AUTO_ROWSTRIDE);
  g_free (src_buf);
  g_free (dst_buf);
}

static void
gegl_op_class_init (GeglOpClass *klass)
{
  GeglOperationClass       *operation_class;
  GeglOperationFilterClass *filter_class;

  operation_class  = GEGL_OPERATION_CLASS (klass);
  filter_class     = GEGL_OPERATION_FILTER_CLASS (klass);

  operation_class->prepare        = prepare;
  operation_class->opencl_support = TRUE;  // Partially enabled, read FIXME
  operation_class->threaded       = FALSE; // XXX: the need for this is a bug

  filter_class->process           = process;

  gegl_operation_class_set_keys (operation_class,
    "name",        "gegl:edge-sobel",
    "title",       _("Sobel Edge Detection"),
    "categories",  "edge-detect",
    "reference-hash", "22e4d815d731db171cef6257c7a95317",
    "description", _("Specialized direction-dependent edge detection"),
          NULL);
}

#endif
