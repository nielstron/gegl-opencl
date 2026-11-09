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
 * Copyright 2001 Spencer Kimball, Bit Specialists, Inc.
 * Copyright 2012 Maxime Nicco <maxime.nicco@gmail.com>
 * Copyright 2013 Téo Mazars <teo.mazars@ensimag.fr>
 */

#include "config.h"
#include <glib/gi18n-lib.h>

#ifdef GEGL_PROPERTIES

property_double (glow_radius, _("Glow radius"), 10.0)
    value_range (1.0, 50.0)
    ui_meta    ("unit", "pixel-distance")

property_double (brightness, _("Brightness"), 0.30)
    value_range (0.0, 1.0)

property_double (sharpness, _("Sharpness"), 0.85)
    value_range (0.0, 1.0)

#else

#define GEGL_OP_AREA_FILTER
#define GEGL_OP_NAME     softglow
#define GEGL_OP_C_SOURCE softglow.c

#include "gegl-op.h"
#include <stdio.h>
#include <stdlib.h>

#define SIGMOIDAL_BASE   2
#define SIGMOIDAL_RANGE  20

static GeglBuffer *
grey_blur_buffer (GeglBuffer          *input,
                  gdouble              glow_radius,
                  const GeglRectangle *result)
{
  GeglNode *gegl, *image, *write, *blur, *crop;
  GeglBuffer *dest;
  gdouble radius, std_dev;

  gegl = gegl_node_new ();
  image = gegl_node_new_child (gegl,
                               "operation", "gegl:buffer-source",
                               "buffer",    input,
                               NULL);

  radius   = fabs (glow_radius) + 1.0;
  std_dev = sqrt (-(radius * radius) / (2 * log (1.0 / 255.0)));

  blur =  gegl_node_new_child (gegl,
                               "operation",    "gegl:gaussian-blur",
                               "std_dev_x",    std_dev,
                               "std_dev_y",    std_dev,
                               "abyss-policy", 0,
                               NULL);

  crop =  gegl_node_new_child (gegl,
                               "operation", "gegl:crop",
                               "x",         (gdouble) result->x,
                               "y",         (gdouble) result->y,
                               "width",     (gdouble) result->width,
                               "height",    (gdouble) result->height,
                               NULL);

  write = gegl_node_new_child (gegl,
                               "operation", "gegl:buffer-sink",
                               "buffer", &dest, NULL);

  gegl_node_link_many (image, blur, crop, write, NULL);
  gegl_node_process (write);

  g_object_unref (gegl);

  return dest;
}

static void
prepare (GeglOperation *operation)
{
  GeglOperationAreaFilter *area = GEGL_OPERATION_AREA_FILTER (operation);
  GeglProperties          *o    = GEGL_PROPERTIES (operation);
  const Babl              *space = gegl_operation_get_source_space (operation, "input");

  area->left = area->right = ceil (fabs (o->glow_radius)) +1;
  area->top = area->bottom = ceil (fabs (o->glow_radius)) +1;

  gegl_operation_set_format (operation, "input",
                             babl_format_with_space ("RGBA float", space));
  gegl_operation_set_format (operation, "output",
                             babl_format_with_space ("RGBA float", space));
}

static GeglRectangle
get_bounding_box (GeglOperation *operation)
{
  GeglRectangle *region;

  region = gegl_operation_source_get_bounding_box (operation, "input");

  if (region != NULL)
    return *region;
  else
    return *GEGL_RECTANGLE (0, 0, 0, 0);
}

#include "opencl/gblur-mem-cl.h"
#include "opencl/softglow.cl.h"

static GeglClRunData *cl_data = NULL;

/* Does what process () does, keeping the blurred image on the GPU */
static gboolean
cl_process (GeglOperation       *operation,
            GeglBuffer          *input,
            GeglBuffer          *output,
            const GeglRectangle *result,
            const GeglRectangle *working_region)
{
  GeglProperties       *o    = GEGL_PROPERTIES (operation);
  const Babl           *rgba = gegl_operation_get_format (operation, "output");
  GeglBufferClIterator *i;
  cl_mem                image = NULL, sigmoid = NULL;
  cl_int                cl_err = CL_SUCCESS;
  cl_int                stride, x0, y0, width;
  cl_float              slope, brightness;
  gboolean              err = FALSE;
  gboolean              ok  = FALSE;
  gdouble               radius, std_dev;
  gint                  read;
  size_t                n, global_ws;

  /* The kernels use the sRGB luminance and TRC */
  if (babl_format_get_space (rgba) != babl_space ("sRGB"))
    return FALSE;

  if (gegl_rectangle_is_empty (working_region))
    return FALSE;

  if (! cl_data)
    {
      const char *kernel_name[] = { "softglow_sigmoid",
                                    "softglow_process",
                                    NULL };

      cl_data = gegl_cl_compile_and_build (softglow_cl_source, kernel_name);
    }

  if (! cl_data)
    return FALSE;

  n = (size_t) working_region->width * working_region->height;

  image = gegl_clCreateBuffer (gegl_cl_get_context (), CL_MEM_READ_WRITE,
                               n * 2 * sizeof (cl_float), NULL, &cl_err);
  if (cl_err != CL_SUCCESS)
    goto cleanup;

  sigmoid = gegl_clCreateBuffer (gegl_cl_get_context (), CL_MEM_READ_WRITE,
                                 n * sizeof (cl_float), NULL, &cl_err);
  if (cl_err != CL_SUCCESS)
    goto cleanup;

  if (! gblur_mem_cl_read (input, working_region, babl_format ("YA float"),
                           image, FALSE))
    goto cleanup;

  slope      = SIGMOIDAL_BASE + (o->sharpness * SIGMOIDAL_RANGE);
  brightness = o->brightness;

  cl_err = gegl_cl_set_kernel_args (cl_data->kernel[0],
                                    sizeof (cl_mem),   &image,
                                    sizeof (cl_mem),   &sigmoid,
                                    sizeof (cl_float), &slope,
                                    sizeof (cl_float), &brightness,
                                    NULL);
  if (cl_err != CL_SUCCESS)
    goto cleanup;

  cl_err = gegl_clEnqueueNDRangeKernel (gegl_cl_get_command_queue (),
                                        cl_data->kernel[0], 1,
                                        NULL, &n, NULL, 0, NULL, NULL);
  if (cl_err != CL_SUCCESS)
    goto cleanup;

  /* Same as grey_blur_buffer (), with the "none" abyss policy */
  radius  = fabs (o->glow_radius) + 1.0;
  std_dev = sqrt (-(radius * radius) / (2 * log (1.0 / 255.0)));

  if (! gblur_mem_cl_blur (sigmoid, sigmoid,
                           working_region->width, working_region->height, 1,
                           std_dev, FALSE))
    goto cleanup;

  stride = working_region->width;

  i = gegl_buffer_cl_iterator_new (output, result, rgba,
                                   GEGL_CL_BUFFER_WRITE);
  read = gegl_buffer_cl_iterator_add (i, input, result, rgba,
                                      GEGL_CL_BUFFER_READ, GEGL_ABYSS_NONE);

  while (gegl_buffer_cl_iterator_next (i, &err) && ! err)
    {
      const GeglRectangle *roi = &i->roi[0];

      x0        = roi->x - working_region->x;
      y0        = roi->y - working_region->y;
      width     = roi->width;
      global_ws = i->size[0];

      cl_err = gegl_cl_set_kernel_args (cl_data->kernel[1],
                                        sizeof (cl_mem), &i->tex[read],
                                        sizeof (cl_mem), &i->tex[0],
                                        sizeof (cl_mem), &sigmoid,
                                        sizeof (cl_int), &stride,
                                        sizeof (cl_int), &x0,
                                        sizeof (cl_int), &y0,
                                        sizeof (cl_int), &width,
                                        NULL);
      if (cl_err == CL_SUCCESS)
        cl_err = gegl_clEnqueueNDRangeKernel (gegl_cl_get_command_queue (),
                                              cl_data->kernel[1], 1,
                                              NULL, &global_ws, NULL,
                                              0, NULL, NULL);
      if (cl_err != CL_SUCCESS)
        {
          gegl_buffer_cl_iterator_stop (i);
          goto cleanup;
        }
    }

  ok = ! err;

cleanup:
  if (sigmoid)
    gegl_clReleaseMemObject (sigmoid);
  if (image)
    gegl_clReleaseMemObject (image);

  return ok;
}

static gboolean
process (GeglOperation       *operation,
         GeglBuffer          *input,
         GeglBuffer          *output,
         const GeglRectangle *result,
         gint                 level)
{
  GeglOperationAreaFilter *area = GEGL_OPERATION_AREA_FILTER (operation);
  GeglProperties          *o    = GEGL_PROPERTIES (operation);
  const Babl              *space = gegl_operation_get_format (operation, "output");

  GeglBuffer *dest, *dest_tmp;

  GeglRectangle  working_region;
  GeglRectangle *whole_region;

  GeglBufferIterator *iter;

  whole_region = gegl_operation_source_get_bounding_box (operation, "input");

  working_region.x      = result->x - area->left;
  working_region.width  = result->width + area->left + area->right;
  working_region.y      = result->y - area->top;
  working_region.height = result->height + area->top + area->bottom;

  gegl_rectangle_intersect (&working_region, &working_region, whole_region);

  if (level == 0 && gegl_operation_use_opencl (operation))
    if (cl_process (operation, input, output, result, &working_region))
      return TRUE;

  dest_tmp = gegl_buffer_new (&working_region, babl_format_with_space ("Y' float", space));

  iter = gegl_buffer_iterator_new (dest_tmp, &working_region, 0, babl_format_with_space ("Y' float", space),
                                   GEGL_ACCESS_WRITE, GEGL_ABYSS_NONE, 2);

  gegl_buffer_iterator_add (iter, input, &working_region, 0, babl_format_with_space ("Y' float", space),
                            GEGL_ACCESS_READ, GEGL_ABYSS_NONE);

  while (gegl_buffer_iterator_next (iter))
    {
      gint    i;
      gfloat *data_out = iter->items[0].data;
      gfloat *data_in  = iter->items[1].data;

      for (i = 0; i < iter->length; i++)
        {
          /* compute sigmoidal transfer */
          gfloat val = *data_in;
          val = 1.0 / (1.0 + exp (-(SIGMOIDAL_BASE + (o->sharpness * SIGMOIDAL_RANGE)) * (val - 0.5)));
          val = val * o->brightness;
          *data_out = CLAMP (val, 0.0, 1.0);

          data_out +=1;
          data_in  +=1;
        }
    }

  dest = grey_blur_buffer (dest_tmp, o->glow_radius, result);

  iter = gegl_buffer_iterator_new (output, result, 0, babl_format_with_space ("RGBA float", space),
                                   GEGL_ACCESS_WRITE, GEGL_ABYSS_NONE, 3);

  gegl_buffer_iterator_add (iter, input, result, 0, babl_format_with_space ("RGBA float", space),
                            GEGL_ACCESS_READ, GEGL_ABYSS_NONE);

  gegl_buffer_iterator_add (iter, dest, result, 0, babl_format_with_space ("Y' float", space),
                            GEGL_ACCESS_READ, GEGL_ABYSS_NONE);

  while (gegl_buffer_iterator_next (iter))
    {
      gint    i;
      gfloat *data_out  = iter->items[0].data;
      gfloat *data_in   = iter->items[1].data;
      gfloat *data_blur = iter->items[2].data;

      for (i = 0; i < iter->length; i++)
        {
          gint c;
          for (c = 0; c < 3; c++)
            {
              gfloat tmp = (1.0 - data_in[c]) * (1.0 - *data_blur);
              data_out[c] = CLAMP (1.0 - tmp, 0.0, 1.0);
            }

          data_out[3] = data_in[3];

          data_out += 4;
          data_in  += 4;
          data_blur+= 1;
        }
    }

  g_object_unref (dest);
  g_object_unref (dest_tmp);

  return TRUE;
}

static void
gegl_op_class_init (GeglOpClass *klass)
{
  GeglOperationClass       *operation_class;
  GeglOperationFilterClass *filter_class;

  operation_class = GEGL_OPERATION_CLASS (klass);
  filter_class    = GEGL_OPERATION_FILTER_CLASS (klass);

  operation_class->prepare          = prepare;
  operation_class->get_bounding_box = get_bounding_box;
  filter_class->process             = process;
  operation_class->threaded         = FALSE;
  operation_class->opencl_support   = TRUE;

  gegl_operation_class_set_keys (operation_class,
    "name",        "gegl:softglow",
    "title",       _("Softglow"),
    "categories",  "artistic",
    "license",     "GPL3+",
    "reference-hash", "04db4506b2e646a5a24e40472407956d",
    "description", _("Simulate glow by making highlights intense and fuzzy"),
    NULL);
}

#endif
