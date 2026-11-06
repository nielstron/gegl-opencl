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
 * Copyright 2007 Mukund Sivaraman <muks@mukund.org>
 */

/* XXX
 * This plug-in isn't really useful apart for compatibility with GIMP
 * scripts, operating with HSV as a color model for filters isn't really
 * useful.
 */

/*
 * The plug-in only does v = 1.0 - v; for each pixel in the image, or
 * each entry in the colormap depending upon the type of image, where 'v'
 * is the value in HSV color model.
 *
 * The plug-in code is optimized towards this, in that it is not a full
 * RGB->HSV->RGB transform, but shortcuts many of the calculations to
 * effectively only do v = 1.0 - v. In fact, hue is never calculated. The
 * shortcuts can be derived from running a set of r, g, b values through the
 * RGB->HSV transform and then from HSV->RGB and solving out the redundant
 * portions.
 *
 */


#include "config.h"
#include <glib/gi18n-lib.h>

#ifdef GEGL_PROPERTIES

   /* no properties */

#else

#define GEGL_OP_POINT_FILTER
#define GEGL_OP_NAME     value_invert
#define GEGL_OP_C_SOURCE value-invert.c

#include "gegl-op.h"

static void
prepare (GeglOperation *operation)
{
  const Babl *space  = gegl_operation_get_source_space (operation, "input");
  const Babl *format = babl_format_with_space ("R'G'B'A float", space);
  const Babl *input  = format;

  /* With OpenCL, read 8-bit input as is: the kernel converts it to float
   * itself, which saves a conversion pass on the device.  The output stays
   * float in any case, so that what follows sees the same data with and
   * without OpenCL.
   */
  if (gegl_operation_use_opencl (operation))
    {
      const Babl *u8 = babl_format_with_space ("R'G'B'A u8", space);

      if (gegl_operation_get_source_format (operation, "input") == u8)
        input = u8;
    }

  gegl_operation_set_format (operation, "input", input);
  gegl_operation_set_format (operation, "output", format);
}

static inline void
value_invert (const gfloat *src,
              gfloat       *dest)
{
  gfloat r, g, b;
  gfloat value, min;
  gfloat delta;

  r = src[0];
  g = src[1];
  b = src[2];

  if (r > g)
    {
      value = MAX (r, b);
      min = MIN (g, b);
    }
  else
    {
      value = MAX (g, b);
      min = MIN (r, b);
    }

  delta = value - min;
  if ((value == 0) || (delta == 0))
    {
      r = 1.0 - value;
      g = 1.0 - value;
      b = 1.0 - value;
    }
  else
    {
      if (r == value)
        {
          r = 1.0 - r;
          b = r * b / value;
          g = r * g / value;
        }
      else if (g == value)
        {
          g = 1.0 - g;
          r = g * r / value;
          b = g * b / value;
        }
      else
        {
          b = 1.0 - b;
          g = b * g / value;
          r = b * r / value;
        }
    }

  dest[0] = r;
  dest[1] = g;
  dest[2] = b;
}

static gboolean
process (GeglOperation       *op,
         void                *in_buf,
         void                *out_buf,
         glong                samples,
         const GeglRectangle *roi,
         gint                 level)
{
  const Babl *format = gegl_operation_get_format (op, "input");
  glong       j;

  if (babl_format_get_type (format, 0) == babl_type ("u8"))
    {
      /* only used when OpenCL processing failed, see prepare () */
      const guint8 *src  = in_buf;
      gfloat       *dest = out_buf;

      for (j = 0; j < samples; j++)
        {
          gfloat rgba[4];
          gint   c;

          for (c = 0; c < 4; c++)
            rgba[c] = src[c] / 255.0f;

          value_invert (rgba, dest);
          dest[3] = rgba[3];

          src  += 4;
          dest += 4;
        }
    }
  else
    {
      const gfloat *src  = in_buf;
      gfloat       *dest = out_buf;

      for (j = 0; j < samples; j++)
        {
          value_invert (src, dest);
          dest[3] = src[3];

          src  += 4;
          dest += 4;
        }
    }

  return TRUE;
}

#include "opencl/gegl-cl.h"
#include "gegl-buffer-cl-iterator.h"
#include "opencl/value-invert.cl.h"

static GeglClRunData *cl_data = NULL;

static gboolean
cl_build (void)
{
  if (!cl_data)
    {
      const char *kernel_name[] = {"gegl_value_invert",
                                   "gegl_value_invert_u8",
                                   "gegl_value_invert_u8_u8",
                                   NULL};
      cl_data = gegl_cl_compile_and_build (value_invert_cl_source, kernel_name);
    }

  return cl_data != NULL;
}

static gboolean
cl_process (GeglOperation       *op,
            cl_mem               in_tex,
            cl_mem               out_tex,
            size_t               global_worksize,
            const GeglRectangle *roi,
            gint                 level)
{
  const Babl *format = gegl_operation_get_format (op, "input");
  cl_kernel   kernel;
  cl_int      cl_err = 0;

  if (!cl_build ())
    return TRUE;

  if (babl_format_get_type (format, 0) == babl_type ("u8"))
    kernel = cl_data->kernel[1];
  else
    kernel = cl_data->kernel[0];

  cl_err = gegl_cl_set_kernel_args (kernel,
                                    sizeof (cl_mem), &in_tex,
                                    sizeof (cl_mem), &out_tex,
                                    NULL);
  CL_CHECK;

  cl_err = gegl_clEnqueueNDRangeKernel (gegl_cl_get_command_queue (),
                                        kernel, 1,
                                        NULL, &global_worksize, NULL,
                                        0, NULL, NULL);
  CL_CHECK;

  return FALSE;

error:
  return TRUE;
}

/* When the output buffer is R'G'B'A u8 itself, for example when rendering
 * straight into an 8-bit blit destination, the float result would only be
 * rounded to u8 by another pass afterwards.  Round it in the kernel
 * instead, the same way, which gives the same result without the float
 * buffer.  What follows the operation still sees float data otherwise.
 */
static gboolean
cl_process_u8 (GeglOperation       *op,
               GeglBuffer          *input,
               GeglBuffer          *output,
               const GeglRectangle *result)
{
  const Babl           *format = gegl_operation_get_format (op, "input");
  GeglBufferClIterator *iter;
  cl_kernel             kernel;
  cl_int                cl_err = 0;
  gboolean              err;

  if (!cl_build ())
    return FALSE;

  kernel = cl_data->kernel[2];

  iter = gegl_buffer_cl_iterator_new (output, result, format,
                                      GEGL_CL_BUFFER_WRITE);
  gegl_buffer_cl_iterator_add (iter, input, result, format,
                               GEGL_CL_BUFFER_READ, GEGL_ABYSS_NONE);

  while (gegl_buffer_cl_iterator_next (iter, &err))
    {
      if (err)
        return FALSE;

      cl_err = gegl_cl_set_kernel_args (kernel,
                                        sizeof (cl_mem), &iter->tex[1],
                                        sizeof (cl_mem), &iter->tex[0],
                                        NULL);
      CL_CHECK;

      cl_err = gegl_clEnqueueNDRangeKernel (gegl_cl_get_command_queue (),
                                            kernel, 1,
                                            NULL, &iter->size[0], NULL,
                                            0, NULL, NULL);
      CL_CHECK;
    }

  return TRUE;

error:
  gegl_buffer_cl_iterator_stop (iter);
  return FALSE;
}

static gboolean
filter_process (GeglOperation       *op,
                GeglBuffer          *input,
                GeglBuffer          *output,
                const GeglRectangle *result,
                gint                 level)
{
  GeglOperationFilterClass *parent_class;
  const Babl               *format = gegl_operation_get_format (op, "input");

  if (gegl_operation_use_opencl (op)                       &&
      babl_format_get_type (format, 0) == babl_type ("u8") &&
      gegl_buffer_get_format (output) == format            &&
      input && input != output                             &&
      result->width > 0 && result->height > 0)
    {
      if (cl_process_u8 (op, input, output, result))
        return TRUE;
    }

  parent_class = GEGL_OPERATION_FILTER_CLASS (gegl_op_parent_class);

  return parent_class->process (op, input, output, result, level);
}

static void
gegl_op_class_init (GeglOpClass *klass)
{
  GeglOperationClass            *operation_class;
  GeglOperationFilterClass      *filter_class;
  GeglOperationPointFilterClass *point_filter_class;

  operation_class    = GEGL_OPERATION_CLASS (klass);
  filter_class       = GEGL_OPERATION_FILTER_CLASS (klass);
  point_filter_class = GEGL_OPERATION_POINT_FILTER_CLASS (klass);

  point_filter_class->process    = process;
  point_filter_class->cl_process = cl_process;
  filter_class->process          = filter_process;
  operation_class->prepare        = prepare;
  operation_class->opencl_support = TRUE;

  gegl_operation_class_set_keys (operation_class,
    "name",        "gegl:value-invert",
    "title",       _("Value Invert"),
    "categories" , "color",
    "reference-hash", "98a6a7c2b289209dc7ce9309063a6796",
    "description",
        _("Invert the value component, the result has the brightness "
          "inverted, keeping the color."),
    NULL);
}

#endif

