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
 * Copyright 2016 Thomas Manni <thomas.manni@free.fr>
 */

 /* compute gradient magnitude and/or direction by central differences */

#include "config.h"
#include <glib/gi18n-lib.h>

#ifdef GEGL_PROPERTIES

enum_start (gegl_imagegradient_output)
   enum_value (GEGL_IMAGEGRADIENT_MAGNITUDE, "magnitude", N_("Magnitude"))
   enum_value (GEGL_IMAGEGRADIENT_DIRECTION, "direction", N_("Direction"))
   enum_value (GEGL_IMAGEGRADIENT_BOTH,      "both",      N_("Both"))
enum_end (GeglImageGradientOutput)

property_enum (output_mode, _("Output mode"),
               GeglImageGradientOutput, gegl_imagegradient_output,
               GEGL_IMAGEGRADIENT_MAGNITUDE)
  description (_("Output Mode"))

#else

#define GEGL_OP_AREA_FILTER
#define GEGL_OP_NAME         image_gradient
#define GEGL_OP_C_SOURCE     image-gradient.c

#define KERNEL_SIZE 1
#define POW2(x) ((x)*(x))

#include "gegl-op.h"
#include "gegl-buffer-cl-iterator.h"
#include "opencl/gegl-cl.h"
#include "opencl/image-gradient.cl.h"

static gboolean
cl_process (GeglBuffer          *input,
            GeglBuffer          *output,
            const GeglRectangle *roi,
            const Babl          *in_format,
            const Babl          *out_format)
{
  static GeglClRunData *cl_data;
  const Babl *buffer_format = gegl_buffer_get_format (input);
  const Babl *output_format = gegl_buffer_get_format (output);
  const Babl *space = babl_format_get_space (in_format);
  GeglBufferClIterator *iter;
  cl_int byte_components = 0;
  cl_int output_ya = FALSE;
  gboolean err;
  cl_int cl_err;
  gint read;

  if (!cl_data)
    {
      const char *kernels[] = { "image_gradient_magnitude", NULL };
      cl_data = gegl_cl_compile_and_build (image_gradient_cl_source, kernels);
    }
  if (!cl_data)
    return FALSE;

  if (buffer_format == babl_format_with_space ("R'G'B'A u8", space) ||
      buffer_format == babl_format_with_space ("R'G'B' u8", space))
    {
      in_format = buffer_format;
      byte_components = babl_format_get_n_components (in_format);
    }

  /* A scalar float result is displayed as opaque linear gray. For the
   * direct sRGB blit, let the iterator quantize gray on the device.
   */
  if (output_format == babl_format ("R'G'B'A u8") &&
      gegl_cl_color_supported (babl_format ("YA float"), output_format) ==
        GEGL_CL_COLOR_CONVERT)
    {
      out_format = babl_format ("YA float");
      output_ya = TRUE;
    }

  iter = gegl_buffer_cl_iterator_new (output, roi, out_format, GEGL_CL_BUFFER_WRITE);
  read = gegl_buffer_cl_iterator_add_2 (iter, input, roi, in_format,
                                        GEGL_CL_BUFFER_READ,
                                        KERNEL_SIZE, KERNEL_SIZE,
                                        KERNEL_SIZE, KERNEL_SIZE,
                                        GEGL_ABYSS_CLAMP);

  while (gegl_buffer_cl_iterator_next (iter, &err))
    {
      const size_t global_ws[] = { iter->roi[0].width, iter->roi[0].height };

      cl_err = gegl_cl_set_kernel_args (cl_data->kernel[0],
                                        sizeof (cl_mem), &iter->tex[read],
                                        sizeof (cl_mem), &iter->tex[0],
                                        sizeof (cl_int), &byte_components,
                                        sizeof (cl_int), &output_ya,
                                        NULL);
      CL_CHECK;

      cl_err = gegl_clEnqueueNDRangeKernel (gegl_cl_get_command_queue (),
                                            cl_data->kernel[0], 2, NULL,
                                            global_ws, NULL, 0, NULL, NULL);
      CL_CHECK;
    }

  return !err;

error:
  gegl_buffer_cl_iterator_stop (iter);
  return FALSE;
}

static void
prepare (GeglOperation *operation)
{
  const Babl *space = gegl_operation_get_source_space (operation, "input");
  GeglOperationAreaFilter *area       = GEGL_OPERATION_AREA_FILTER (operation);
  GeglProperties          *o          = GEGL_PROPERTIES (operation);
  const Babl              *rgb_format = babl_format_with_space ("R'G'B' float", space);
  const Babl              *out_format = babl_format_n (babl_type ("float"), 2);

  area->left   =
  area->top    =
  area->right  =
  area->bottom = KERNEL_SIZE;

  if (o->output_mode == GEGL_IMAGEGRADIENT_MAGNITUDE ||
      o->output_mode == GEGL_IMAGEGRADIENT_DIRECTION)
    {
      out_format = babl_format_n (babl_type ("float"), 1);
    }

  gegl_operation_set_format (operation, "input",  rgb_format);
  gegl_operation_set_format (operation, "output", out_format);
}

static GeglRectangle
get_bounding_box (GeglOperation *operation)
{
  GeglRectangle  result = { 0, 0, 0, 0 };
  GeglRectangle *in_rect;

  in_rect = gegl_operation_source_get_bounding_box (operation, "input");
  if (in_rect)
    {
      result = *in_rect;
    }

  return result;
}

static gboolean
process (GeglOperation       *operation,
         GeglBuffer          *input,
         GeglBuffer          *output,
         const GeglRectangle *roi,
         gint                 level)
{
  const Babl     *in_format      = gegl_operation_get_format (operation, "input");
  const Babl     *out_format     = gegl_operation_get_format (operation, "output");
  const gint      in_components  = babl_format_get_n_components (in_format);
  const gint      out_components = babl_format_get_n_components (out_format);
  GeglProperties *o              = GEGL_PROPERTIES (operation);
  GeglRectangle   in_roi         = {
    roi->x - KERNEL_SIZE,
    roi->y - KERNEL_SIZE,
    roi->width  + KERNEL_SIZE * 2,
    roi->height + KERNEL_SIZE * 2,
  };

  /* Direction depends on the winning channel, which can change when two
   * rounded magnitudes tie. Keep those modes on the CPU.
   */
  if (level == 0 && o->output_mode == GEGL_IMAGEGRADIENT_MAGNITUDE &&
      gegl_operation_use_opencl (operation) &&
      cl_process (input, output, roi, in_format, out_format))
    return TRUE;

  GeglBufferIterator *iter = gegl_buffer_iterator_new (
      output, roi, 0, out_format, GEGL_ACCESS_WRITE, GEGL_ABYSS_NONE, 2);
  gegl_buffer_iterator_add (iter, input, &in_roi, 0, in_format, GEGL_ACCESS_READ,
                            GEGL_ABYSS_CLAMP);

  while (gegl_buffer_iterator_next (iter))
    {
      const gfloat *in  = iter->items[1].data;
      gfloat       *out = iter->items[0].data;

      for (gint y = 0; y < iter->items[0].roi.height; y++)
        {
          gint y_offset = (y + KERNEL_SIZE) * iter->items[1].roi.width;

          for (gint x = 0; x < iter->items[0].roi.width; x++)
            {
              gfloat dx[3];
              gfloat dy[3];
              gfloat magnitude[3];
              for (gint j = 0; j < in_components; j++)
                {
                  #define KERNEL(dx, dy, c) in[((y_offset + (dy * iter->items[1].roi.width)) + \
                                                (KERNEL_SIZE + x + dx)) * in_components + c]
                  dx[j] = KERNEL(-1, 0, j) - KERNEL(1, 0, j);
                  dy[j] = KERNEL(0, -1, j) - KERNEL(0, 1, j);
                  #undef KERNEL

                  magnitude[j] = sqrtf(POW2(dx[j]) + POW2(dy[j]));
              };

              gint max_index = 0;
              if (magnitude[0] <= magnitude[1])
                max_index = 1;
              if (magnitude[max_index] < magnitude[2])
                max_index = 2;

              if (o->output_mode == GEGL_IMAGEGRADIENT_MAGNITUDE ||
                  o->output_mode == GEGL_IMAGEGRADIENT_BOTH)
                {
                  *out = magnitude[max_index];
                }

              if (o->output_mode == GEGL_IMAGEGRADIENT_DIRECTION ||
                  o->output_mode == GEGL_IMAGEGRADIENT_BOTH)
                {
                  gfloat direction = atan2 (dy[max_index], dx[max_index]);

                  if (o->output_mode == GEGL_IMAGEGRADIENT_DIRECTION)
                    *out = direction;
                  else
                    *(out + 1) = direction;
                }

              out += out_components;
            }
        }
    }

  return TRUE;
}

static void
gegl_op_class_init (GeglOpClass *klass)
{
  GeglOperationClass       *operation_class = GEGL_OPERATION_CLASS (klass);
  GeglOperationFilterClass *filter_class    = GEGL_OPERATION_FILTER_CLASS (klass);

  operation_class->prepare          = prepare;
  operation_class->get_bounding_box = get_bounding_box;
  operation_class->opencl_support   = TRUE;

  filter_class->process = process;

  gegl_operation_class_set_keys (operation_class,
    "name",        "gegl:image-gradient",
    "title",       _("Image Gradient"),
    "categories",  "edge-detect",
    "reference-hash", "3bc1f4413a06969bf86606d621969651",
    "description", _("Compute gradient magnitude and/or direction by "
                     "central differences"),
    NULL);
}

#endif
