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
 * Copyright (C) 1997 Andrew Kieschnick (andrewk@mail.utexas.edu)
 *
 * Original deinterlace for GIMP 0.54 API by Federico Mena Quintero
 *
 * Copyright (C) 1996 Federico Mena Quintero
 *
 * Copyright (C) 2011 Robert Sasu <sasu.robert@gmail.com>
 */

#include "config.h"
#include <glib/gi18n-lib.h>

#ifdef GEGL_PROPERTIES

enum_start (gegl_deinterlace_keep)
  enum_value (GEGL_DEINTERLACE_KEEP_EVEN, "even", N_("Keep even fields"))
  enum_value (GEGL_DEINTERLACE_KEEP_ODD,  "odd",  N_("Keep odd fields"))
enum_end (GeglDeinterlaceKeep)

property_enum (keep, _("Keep"),
  GeglDeinterlaceKeep, gegl_deinterlace_keep,
  GEGL_DEINTERLACE_KEEP_EVEN)
  description(_("Keep even or odd fields"))

property_enum (orientation, _("Orientation"),
  GeglOrientation, gegl_orientation,
  GEGL_ORIENTATION_HORIZONTAL)
  description(_("Deinterlace horizontally or vertically"))

property_int  (size, _("Block size"), 1)
  value_range (0, 100)
  description (_("Block size of deinterlacing rows/columns"))

#else

#define GEGL_OP_AREA_FILTER
#define GEGL_OP_NAME     deinterlace
#define GEGL_OP_C_SOURCE deinterlace.c

#include "gegl-op.h"
#include <stdio.h>

static void
prepare (GeglOperation *operation)
{
  GeglOperationAreaFilter *op_area = GEGL_OPERATION_AREA_FILTER (operation);
  GeglProperties          *o       = GEGL_PROPERTIES (operation);
  const Babl              *space   = gegl_operation_get_source_space (operation, "input");

  if (o->orientation == GEGL_ORIENTATION_HORIZONTAL)
    {
      op_area->left = op_area->right = 0;
      op_area->top = op_area->bottom = o->size + 1;
    }
  else
    {
      op_area->left = op_area->right = o->size + 1;
      op_area->top = op_area->bottom = 0;
    }

  gegl_operation_set_format (operation, "input",
                             babl_format_with_space ("RGBA float", space));

  /* The OpenCL path applies the sRGB TRC itself, since R'G'B'A float is
   * much cheaper to convert to the usual 8-bit formats on the CPU.
   */
  if (gegl_operation_use_opencl (operation) &&
      babl_format_get_space (babl_format_with_space ("RGBA float", space)) ==
      babl_space ("sRGB"))
    gegl_operation_set_format (operation, "output",
                               babl_format ("R'G'B'A float"));
  else
    gegl_operation_set_format (operation, "output",
                               babl_format_with_space ("RGBA float", space));
}

static GeglRectangle
get_bounding_box (GeglOperation *operation)
{
  GeglRectangle  result = { 0, 0, 0, 0 };
  GeglRectangle *in_rect;

  in_rect = gegl_operation_source_get_bounding_box (operation, "input");
  if (!in_rect)
    return result;

  return *in_rect;
}

static void
deinterlace_horizontal (gfloat              *src_buf,
                        gfloat              *dest,
                        const GeglRectangle *result,
                        const GeglRectangle *extended,
                        const GeglRectangle *boundary,
                        gint                 inter,
                        gint                 y,
                        gint                 size)
{
  gfloat upper[4];
  gfloat lower[4];
  gfloat temp_buf[4];
  gint   x;
  gint   up_offset;
  gint   low_offset;
  gint   offset = 0;
  gint   i;

  for (x=0; x < result->width; x++)
    {
      gfloat ualpha, lalpha, temp;
      gfloat alpha = 0;

      temp_buf[0] = temp_buf[1] = temp_buf[2] = temp_buf[3] = 0;

      for (i = 0; i < size; i++)
        {
          gint b;

          if (y  - i> 0)
            up_offset = (y - i - extended->y) * extended->width * 4;
          else
            up_offset = inter * extended->width * 4;

          if (y + i + 1 < boundary->height)
            low_offset = (y + i + 1 - extended->y) * extended->width * 4;
          else
            low_offset = (y - 1 + inter - extended->y) * extended->width * 4;

          offset = (y - result->y) * extended->width * 4;

          for (b=0; b<4; b++)
            {
              upper[b] = src_buf[up_offset + x * 4 + b];
              lower[b] = src_buf[low_offset + x * 4 + b];
            }

          ualpha = upper[3];
          lalpha = lower[3];
          temp   = ualpha + lalpha;
          alpha += temp;

          for (b=0; b < 3; b++)
            temp_buf[b] += (upper[b] * ualpha +
                            lower[b] * lalpha);
        }

      if ((dest[offset + x * 4 + 3] = alpha / (2 * size)))
        {
          gint b;
          for (b=0; b < 3; b++)
            dest[offset + x * 4 + b] = temp_buf[b] / alpha;

        }
    }
}

static void
deinterlace_vertical (gfloat              *src_buf,
                      gfloat              *dest,
                      const GeglRectangle *result,
                      const GeglRectangle *extended,
                      const GeglRectangle *boundary,
                      gint                 inter,
                      gint                 x,
                      gint                 size)
{
  gfloat upper[4];
  gfloat lower[4];
  gfloat temp_buf[4];
  gint   y;
  gint   up_offset;
  gint   low_offset;
  gint   offset = 0;
  gint   i;

  for (y=result->y; y < result->y + result->height; y++)
    {
      gfloat ualpha, lalpha, temp;
      gfloat alpha = 0;

      temp_buf[0] = temp_buf[1] = temp_buf[2] = temp_buf[3] = 0;

      for (i = 0; i < size; i++)
        {
          gint b;

          if (x  - i > 0)
            up_offset = (y - extended->y) * extended->width * 4
              + (x - i - extended->x) * 4;
          else
            up_offset = (y - extended->y) * extended->width * 4 + inter * 4;

          if (x + i + 1 < boundary->width)
            low_offset = (y - extended->y) * extended->width * 4 +
              (x + i + 1 - extended->x) * 4;
          else
            low_offset = (y - extended->y) * extended->width * 4 +
              (x + i - 1 + inter - extended->x) * 4;

          offset = (y - result->y) * result->width * 4 + (x - result->x) * 4;

          for (b=0; b<4; b++)
            {
              upper[b] = src_buf[up_offset + b];
              lower[b] = src_buf[low_offset + b];
            }

          ualpha = upper[3];
          lalpha = lower[3];
          temp   = ualpha + lalpha;
          alpha += temp;

          for (b=0; b < 3; b++)
            temp_buf[b] += (upper[b] * ualpha +
                            lower[b] * lalpha);
        }

      if ((dest[offset + 3] = alpha / (2 * size)))
        {
          gint b;
          for (b=0; b < 3; b++)
            dest[offset + b] = temp_buf[b] / alpha;
        }
    }
}

#include "opencl/gegl-cl.h"
#include "gegl-buffer-cl-iterator.h"
#include "opencl/deinterlace.cl.h"

static GeglClRunData *cl_data = NULL;

static gboolean
cl_process (GeglOperation       *operation,
            GeglBuffer          *input,
            GeglBuffer          *output,
            const GeglRectangle *result,
            const GeglRectangle *boundary)
{
  GeglProperties          *o          = GEGL_PROPERTIES (operation);
  GeglOperationAreaFilter *op_area    = GEGL_OPERATION_AREA_FILTER (operation);
  const Babl              *in_format  = gegl_operation_get_format (operation, "input");
  const Babl              *out_format = gegl_operation_get_format (operation, "output");
  GeglBufferClIterator    *i;
  cl_int                   horizontal = o->orientation == GEGL_ORIENTATION_HORIZONTAL;
  cl_int                   keep       = o->keep;
  cl_int                   size       = o->size;
  cl_int                   to_gamma;
  cl_int                   extent;
  cl_int                   rect_origin;
  cl_int                   rect_x;
  cl_int                   rect_width;
  cl_int                   wrap_y     = result->y;
  cl_mem                   wrap       = NULL;
  cl_int                   cl_err     = 0;
  gboolean                 err        = FALSE;
  gint                     read;

  /* The CPU code mixes absolute and relative coordinates in a way that
   * only works for an input at the origin; size 0 divides by zero.
   */
  if (boundary->x != 0 || boundary->y != 0 || size < 1)
    return FALSE;

  if (out_format == in_format)
    to_gamma = FALSE;
  else if (babl_format_get_space (in_format) == babl_space ("sRGB") &&
           babl_model_is (babl_format_get_model (out_format), "R'G'B'A"))
    to_gamma = TRUE;
  else
    return FALSE;

  /* The source rectangle of process () */
  rect_x     = CLAMP (result->x - op_area->left, boundary->x,
                      boundary->x + boundary->width);
  rect_width = CLAMP (result->width + op_area->left + op_area->right, 0,
                      boundary->width);

  if (horizontal)
    {
      extent      = boundary->height;
      rect_origin = CLAMP (result->y - op_area->top, boundary->y,
                           boundary->y + boundary->height);
    }
  else
    {
      extent      = boundary->width;
      rect_origin = rect_x;
    }

  if (rect_width < 1)
    return FALSE;

  /* In the vertical orientation the CPU code reads up to size - 1
   * columns past the end of a row of its source rectangle, which then
   * come from the start of the next row (see below). Only when the
   * rectangle is wider than size do they all stay within that next row,
   * and does the CPU code not read column -1 (a width of 1 with keep
   * even); this only excludes images at most size pixels wide.
   */
  if (!horizontal && rect_width <= size)
    return FALSE;

  if (!cl_data)
    {
      const char *kernel_name[] = { "deinterlace", NULL };
      cl_data = gegl_cl_compile_and_build (deinterlace_cl_source, kernel_name);
    }

  if (!cl_data)
    return FALSE;

  /* In the vertical orientation, the CPU code reads columns past the end
   * of a row of its source rectangle from the start of the next row; the
   * OpenCL chunks need not contain these, so upload them separately (one
   * row more than the CPU has, where it reads past its buffer).
   */
  {
    GeglRectangle  wrap_rect = { rect_x, result->y, size, result->height + 1 };
    gint           n         = horizontal ? 1 : wrap_rect.width * wrap_rect.height;
    gfloat        *wrap_buf  = g_new0 (gfloat, n * 4);

    if (!horizontal)
      gegl_buffer_get (input, &wrap_rect, 1.0, in_format, wrap_buf,
                       GEGL_AUTO_ROWSTRIDE, GEGL_ABYSS_NONE);

    wrap = gegl_clCreateBuffer (gegl_cl_get_context (),
                                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                n * 4 * sizeof (gfloat), wrap_buf, &cl_err);
    g_free (wrap_buf);

    if (cl_err != CL_SUCCESS)
      return FALSE;
  }

  i = gegl_buffer_cl_iterator_new (output, result, out_format,
                                   GEGL_CL_BUFFER_WRITE);

  read = gegl_buffer_cl_iterator_add_2 (i, input, result, in_format,
                                        GEGL_CL_BUFFER_READ,
                                        op_area->left, op_area->right,
                                        op_area->top,  op_area->bottom,
                                        GEGL_ABYSS_NONE);

  while (gegl_buffer_cl_iterator_next (i, &err) && !err)
    {
      const GeglRectangle *roi = &i->roi[0];
      size_t  global_ws[2] = { roi->width, roi->height };
      cl_int  roi_x        = roi->x;
      cl_int  roi_y        = roi->y;
      cl_int  in_width     = i->roi[read].width;
      cl_int  in_x         = i->roi[read].x;
      cl_int  in_y         = i->roi[read].y;

      cl_err = gegl_cl_set_kernel_args (cl_data->kernel[0],
                                        sizeof (cl_mem), &i->tex[read],
                                        sizeof (cl_mem), &wrap,
                                        sizeof (cl_mem), &i->tex[0],
                                        sizeof (cl_int), &horizontal,
                                        sizeof (cl_int), &keep,
                                        sizeof (cl_int), &size,
                                        sizeof (cl_int), &roi_x,
                                        sizeof (cl_int), &roi_y,
                                        sizeof (cl_int), &in_width,
                                        sizeof (cl_int), &in_x,
                                        sizeof (cl_int), &in_y,
                                        sizeof (cl_int), &extent,
                                        sizeof (cl_int), &rect_origin,
                                        sizeof (cl_int), &rect_x,
                                        sizeof (cl_int), &rect_width,
                                        sizeof (cl_int), &wrap_y,
                                        sizeof (cl_int), &to_gamma,
                                        NULL);
      CL_CHECK;

      cl_err = gegl_clEnqueueNDRangeKernel (gegl_cl_get_command_queue (),
                                            cl_data->kernel[0], 2,
                                            NULL, global_ws, NULL,
                                            0, NULL, NULL);
      CL_CHECK;
    }

  gegl_clReleaseMemObject (wrap);

  return !err;

error:
  gegl_buffer_cl_iterator_stop (i);
  gegl_clReleaseMemObject (wrap);

  return FALSE;
}

static gboolean
process (GeglOperation       *operation,
         GeglBuffer          *input,
         GeglBuffer          *output,
         const GeglRectangle *result,
         gint                 level)
{
  GeglProperties          *o        = GEGL_PROPERTIES (operation);
  GeglOperationAreaFilter *op_area  = GEGL_OPERATION_AREA_FILTER (operation);
  const Babl              *format   = gegl_operation_get_format (operation, "input");
  GeglRectangle            rect;
  GeglRectangle            boundary = get_bounding_box (operation);
  gint                     x, y;
  gfloat                  *dst_buf, *src_buf;

  if (gegl_operation_use_opencl (operation) &&
      cl_process (operation, input, output, result, &boundary))
    return TRUE;

  rect.x      = CLAMP (result->x - op_area->left, boundary.x, boundary.x +
                       boundary.width);
  rect.width  = CLAMP (result->width + op_area->left + op_area->right, 0,
                       boundary.width);
  rect.y      = CLAMP (result->y - op_area->top, boundary.y, boundary.y +
                       boundary.height);
  rect.height = CLAMP (result->height + op_area->top + op_area->bottom, 0,
                       boundary.height);

  dst_buf = g_new0 (gfloat, result->height * result->width * 4);
  src_buf = g_new0 (gfloat, rect.height * rect.width * 4);

  gegl_buffer_get (input, result, 1.0, format, dst_buf,
                   GEGL_AUTO_ROWSTRIDE, GEGL_ABYSS_NONE);
  gegl_buffer_get (input, &rect, 1.0, format, src_buf,
                   GEGL_AUTO_ROWSTRIDE, GEGL_ABYSS_NONE);

  if (o->orientation == GEGL_ORIENTATION_HORIZONTAL)
    {
      for (y = result->y; y < result->y + result->height; y++)
        if ((o->keep == GEGL_DEINTERLACE_KEEP_EVEN && (y % 2 == 0)) ||
            (o->keep == GEGL_DEINTERLACE_KEEP_ODD  && (y % 2 != 0)))
          {
            deinterlace_horizontal (src_buf, dst_buf, result, &rect, &boundary,
                                    o->keep,
                                    y, o->size);
          }
    }
  else
    {
      for (x = result->x; x < result->x + result->width; x++)
        if ((o->keep == GEGL_DEINTERLACE_KEEP_EVEN && (x % 2 == 0)) ||
            (o->keep == GEGL_DEINTERLACE_KEEP_ODD  && (x % 2 != 0)))
          {
            deinterlace_vertical (src_buf, dst_buf, result, &rect, &boundary,
                                  o->keep,
                                  x, o->size);
          }
    }

  gegl_buffer_set (output, result, 0, format, dst_buf, GEGL_AUTO_ROWSTRIDE);

  g_free (src_buf);
  g_free (dst_buf);

  return  TRUE;
}

static void
gegl_op_class_init (GeglOpClass *klass)
{
  GeglOperationClass       *operation_class;
  GeglOperationFilterClass *filter_class;

  operation_class = GEGL_OPERATION_CLASS (klass);
  filter_class    = GEGL_OPERATION_FILTER_CLASS (klass);

  operation_class->prepare           = prepare;
  operation_class->get_bounding_box  = get_bounding_box;
  filter_class->process              = process;
  operation_class->opencl_support    = TRUE;

  gegl_operation_class_set_keys (operation_class,
    "name",               "gegl:deinterlace",
    "title",              _("Deinterlace"),
    "categories",         "enhance",
    "license",            "GPL3+",
    "position-dependent", "true",
    "reference-hash",     "c1b3b9cdb55e737e6282a90603df9755",
    "description", _("Fix images where every other row or column is missing"),
    NULL);
}

#endif
