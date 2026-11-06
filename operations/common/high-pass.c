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
 * Copyright 2013 Ville Sokk <ville.sokk@gmail.com>
 */

#include "config.h"
#include <glib/gi18n-lib.h>

#ifdef GEGL_PROPERTIES

property_double (std_dev, _("Standard deviation"), 4.0)
    description (_("Standard deviation (spatial scale factor)"))
    value_range (0.0, 10000.0)
    ui_range (0.0, 1000.0)
    ui_gamma (1.5)

property_double (contrast, _("Contrast"), 1.0)
    description(_("Contrast of high-pass"))
    value_range (0.0, 5.0)
    ui_range (0.0, 5.0)

#else

#define GEGL_OP_FILTER
#define GEGL_OP_NAME     high_pass
#define GEGL_OP_C_SOURCE high-pass.c

#include "gegl-op.h"

/* The high-pass is computed by the graph below. With OpenCL the whole
 * graph runs in one go on the GPU instead, see cl_process ().
 */
static GeglNode *
build_graph (GeglNode       *gegl,
             GeglBuffer     *input,
             GeglProperties *o)
{
  GeglNode *image, *invert, *blur, *opacity, *over, *sub, *mul, *add, *gam0, *gam1;

  image = gegl_node_new_child (gegl, "operation", "gegl:buffer-source",
                               "buffer", input, NULL);
  invert = gegl_node_new_child (gegl, "operation", "gegl:invert-gamma", NULL);
  blur = gegl_node_new_child (gegl, "operation", "gegl:gaussian-blur",
                              "std-dev-x", o->std_dev,
                              "std-dev-y", o->std_dev,
                              NULL);
  opacity = gegl_node_new_child (gegl, "operation", "gegl:opacity",
                                 "value", 0.5, NULL);
  over = gegl_node_new_child (gegl, "operation", "gegl:over", "srgb", TRUE, NULL);
  gam0 = gegl_node_new_child (gegl, "operation", "gegl:gamma", "value", 0.4545, NULL);
  sub = gegl_node_new_child (gegl, "operation", "gegl:add", "value", -0.5, NULL);
  mul = gegl_node_new_child (gegl, "operation", "gegl:multiply",
                             "value", o->contrast, NULL);
  add = gegl_node_new_child (gegl, "operation", "gegl:add", "value", 0.5, NULL);
  gam1 = gegl_node_new_child (gegl, "operation", "gegl:gamma", "value", 2.2, NULL);

  gegl_node_link_many (image, blur, invert, opacity, NULL);
  gegl_node_connect (opacity, "output", over, "aux");
  gegl_node_link_many (image, over, gam0, sub, mul, add, gam1, NULL);

  return gam1;
}

static void
prepare (GeglOperation *operation)
{
  const Babl *format = gegl_operation_get_source_format (operation, "input");

  gegl_operation_set_format (operation, "input", format);

  /* The format the graph produces: that of gegl:gamma after gegl:over */
  format = gegl_babl_variant (format,
                              GEGL_BABL_VARIANT_PERCEPTUAL_PREMULTIPLIED);
  format = gegl_babl_variant (format, GEGL_BABL_VARIANT_LINEAR);

  gegl_operation_set_format (operation, "output", format);
}

/* The gaussian blur needs whole lines of the input */
static GeglRectangle
get_required_for_output (GeglOperation       *operation,
                         const gchar         *input_pad,
                         const GeglRectangle *roi)
{
  const GeglRectangle *in_rect =
    gegl_operation_source_get_bounding_box (operation, "input");

  if (in_rect && ! gegl_rectangle_is_infinite_plane (in_rect))
    return *in_rect;

  return *roi;
}

static GeglRectangle
get_cached_region (GeglOperation       *operation,
                   const GeglRectangle *roi)
{
  return get_required_for_output (operation, "input", roi);
}

#include "opencl/gblur-mem-cl.h"
#include "opencl/high-pass.cl.h"

static GeglClRunData *cl_data = NULL;

static gboolean
cl_process (GeglOperation       *operation,
            GeglBuffer          *input,
            GeglBuffer          *output,
            const GeglRectangle *result)
{
  GeglProperties       *o      = GEGL_PROPERTIES (operation);
  const GeglRectangle  *extent = gegl_buffer_get_extent (input);
  const Babl           *ragaba = babl_format ("RaGaBaA float");
  const Babl           *rgba   = babl_format ("RGBA float");
  GeglBufferClIterator *i;
  cl_mem                image = NULL, blur = NULL;
  cl_int                cl_err = CL_SUCCESS;
  cl_int                stride, x0, y0, width;
  cl_float              contrast;
  gboolean              err = FALSE;
  gboolean              ok  = FALSE;
  size_t                n, global_ws;

  /* The kernel computes the graph for sRGB RGB(A) pixels */
  if (gegl_operation_get_format (operation, "output") != rgba ||
      babl_format_get_space (gegl_buffer_get_format (input)) !=
      babl_space ("sRGB"))
    return FALSE;

  if (gegl_rectangle_is_empty (result) ||
      ! gegl_rectangle_contains (extent, result))
    return FALSE;

  if (! cl_data)
    {
      const char *kernel_name[] = { "high_pass", NULL };

      cl_data = gegl_cl_compile_and_build (high_pass_cl_source, kernel_name);
    }

  if (! cl_data)
    return FALSE;

  n = (size_t) extent->width * extent->height;

  image = gegl_clCreateBuffer (gegl_cl_get_context (), CL_MEM_READ_WRITE,
                               n * 4 * sizeof (cl_float), NULL, &cl_err);
  if (cl_err != CL_SUCCESS)
    goto cleanup;

  blur = gegl_clCreateBuffer (gegl_cl_get_context (), CL_MEM_READ_WRITE,
                              n * 4 * sizeof (cl_float), NULL, &cl_err);
  if (cl_err != CL_SUCCESS)
    goto cleanup;

  /* gegl:gaussian-blur works in RaGaBaA float */
  if (! gblur_mem_cl_read (input, extent, ragaba, image, FALSE) ||
      ! gblur_mem_cl_blur (image, blur, extent->width, extent->height, 4,
                           o->std_dev, TRUE))
    goto cleanup;

  stride   = extent->width;
  contrast = o->contrast;

  i = gegl_buffer_cl_iterator_new (output, result, rgba,
                                   GEGL_CL_BUFFER_WRITE);

  while (gegl_buffer_cl_iterator_next (i, &err) && ! err)
    {
      const GeglRectangle *roi = &i->roi[0];

      x0        = roi->x - extent->x;
      y0        = roi->y - extent->y;
      width     = roi->width;
      global_ws = i->size[0];

      cl_err = gegl_cl_set_kernel_args (cl_data->kernel[0],
                                        sizeof (cl_mem),   &image,
                                        sizeof (cl_mem),   &blur,
                                        sizeof (cl_mem),   &i->tex[0],
                                        sizeof (cl_int),   &stride,
                                        sizeof (cl_int),   &x0,
                                        sizeof (cl_int),   &y0,
                                        sizeof (cl_int),   &width,
                                        sizeof (cl_float), &contrast,
                                        NULL);
      if (cl_err == CL_SUCCESS)
        cl_err = gegl_clEnqueueNDRangeKernel (gegl_cl_get_command_queue (),
                                              cl_data->kernel[0], 1,
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
  if (blur)
    gegl_clReleaseMemObject (blur);
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
  GeglProperties      *o       = GEGL_PROPERTIES (operation);
  const GeglRectangle *in_rect =
    gegl_operation_source_get_bounding_box (operation, "input");
  GeglBuffer          *source;
  GeglNode            *gegl;
  GeglNode            *sink;

  /* Clip the input to its bounding box, which is where the blur clamps */
  if (in_rect && ! gegl_rectangle_is_infinite_plane (in_rect))
    source = gegl_buffer_create_sub_buffer (input, in_rect);
  else
    source = g_object_ref (input);

  if (level == 0 && gegl_operation_use_opencl (operation) &&
      cl_process (operation, source, output, result))
    {
      g_object_unref (source);
      return TRUE;
    }

  gegl = gegl_node_new ();
  g_object_set (gegl, "use-opencl", gegl_operation_use_opencl (operation), NULL);

  sink = build_graph (gegl, source, o);
  gegl_node_blit_buffer (sink, output, result, level, GEGL_ABYSS_NONE);

  g_object_unref (gegl);
  g_object_unref (source);

  return TRUE;
}

static void
gegl_op_class_init (GeglOpClass *klass)
{
  GeglOperationClass       *operation_class;
  GeglOperationFilterClass *filter_class;

  operation_class = GEGL_OPERATION_CLASS (klass);
  filter_class    = GEGL_OPERATION_FILTER_CLASS (klass);

  operation_class->prepare                 = prepare;
  operation_class->get_required_for_output = get_required_for_output;
  operation_class->get_cached_region       = get_cached_region;
  operation_class->threaded                = FALSE;
  operation_class->opencl_support          = TRUE;
  filter_class->process                    = process;

  gegl_operation_class_set_keys (operation_class,
                                 "name",        "gegl:high-pass",
                                 "title",       _("High Pass Filter"),
                                 "categories",  "frequency",
                                 "reference-hash", "cf04bb714a173f1f71da8af283b8a894",
                                 "description",
                                 _("Enhances fine details."),
                                 NULL);
}

#endif
