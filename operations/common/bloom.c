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
 * Copyright 2020 Ell
 */

#include "config.h"
#include <glib/gi18n-lib.h>

#define EPSILON 1e-6

#ifdef GEGL_PROPERTIES

property_double (threshold, _("Threshold"), 50.0)
    description (_("Glow-area brightness threshold"))
    ui_range    (0.0, 100.0)

property_double (softness, _("Softness"), 25.0)
    description (_("Glow-area edge softness"))
    value_range (0.0, G_MAXDOUBLE)
    ui_range    (0.0, 100.0)

property_double (radius, _("Radius"), 10.0)
    description (_("Glow radius"))
    value_range (0.0, 1500.0)
    ui_range    (0.0, 100.0)
    ui_gamma    (2.0)
    ui_meta     ("unit", "pixel-distance")

property_double (strength, _("Strength"), 50.0)
    description (_("Glow strength"))
    value_range (0.0, G_MAXDOUBLE)
    ui_range    (0.0, 100.0)

property_boolean (limit_exposure, _("Limit exposure"), FALSE)
    description (_("Don't over-expose highlights"))

#else

#define GEGL_OP_FILTER
#define GEGL_OP_NAME     bloom
#define GEGL_OP_C_SOURCE bloom.c

#include "gegl-op.h"

/* The bloom is computed by the graph below. With OpenCL the whole graph
 * runs in one go on the GPU instead, see cl_process ().
 */
static GeglNode *
build_graph (GeglNode       *gegl,
             GeglBuffer     *input,
             GeglProperties *o)
{
  GeglNode *image, *convert_format, *cast_format, *levels, *rgb_clip;
  GeglNode *multiply, *gaussian_blur, *combine;

  image = gegl_node_new_child (
    gegl,
    "operation",     "gegl:buffer-source",
    "buffer",        input,
    NULL);

  convert_format = gegl_node_new_child (
    gegl,
    "operation",     "gegl:convert-format",
    "format",        babl_format ("Y float"),
    NULL);

  cast_format    = gegl_node_new_child (
    gegl,
    "operation",     "gegl:cast-format",
    "input-format",  babl_format ("Y' float"),
    "output-format", babl_format ("Y float"),
    NULL);

  levels         = gegl_node_new_child (
    gegl,
    "operation",     "gegl:levels",
    "in-low",        (o->threshold - o->softness) / 100.0,
    "in-high",       (o->threshold + o->softness) / 100.0,
    "out-high",      o->strength / 100.0,
    NULL);

  rgb_clip       = gegl_node_new_child (
    gegl,
    "operation",     "gegl:rgb-clip",
    "high-limit",    o->strength / 100.0,
    NULL);

  multiply       = gegl_node_new_child (
    gegl,
    "operation",     "gegl:multiply",
    NULL);

  gaussian_blur  = gegl_node_new_child (
    gegl,
    "operation",     "gegl:gaussian-blur",
    "std-dev-x",     o->radius,
    "std-dev-y",     o->radius,
    NULL);

  combine        = gegl_node_new_child (
    gegl,
    "operation",     o->limit_exposure ? "gegl:screen" : "gegl:add",
    NULL);

  gegl_node_link_many (image, convert_format, cast_format, levels, rgb_clip,
                       NULL);

  gegl_node_connect (image, "output", multiply, "input");
  gegl_node_connect (rgb_clip, "output", multiply, "aux");

  gegl_node_link (multiply, gaussian_blur);

  gegl_node_connect (image, "output", combine, "input");
  gegl_node_connect (gaussian_blur, "output", combine, "aux");

  return combine;
}

static void
prepare (GeglOperation *operation)
{
  GeglProperties *o      = GEGL_PROPERTIES (operation);
  const Babl     *format = gegl_operation_get_source_format (operation,
                                                             "input");

  gegl_operation_set_format (operation, "input", format);

  /* The format the graph produces: that of gegl:add or gegl:screen */
  format = gegl_babl_variant (format,
                              o->limit_exposure ?
                                GEGL_BABL_VARIANT_LINEAR_PREMULTIPLIED :
                                GEGL_BABL_VARIANT_LINEAR);

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
#include "opencl/bloom.cl.h"

static GeglClRunData *cl_data = NULL;

static gboolean
cl_process (GeglOperation       *operation,
            GeglBuffer          *input,
            GeglBuffer          *output,
            const GeglRectangle *result)
{
  GeglProperties       *o      = GEGL_PROPERTIES (operation);
  const GeglRectangle  *extent = gegl_buffer_get_extent (input);
  const Babl           *rgba   = babl_format ("RGBA float");
  const Babl           *out_format;
  GeglBufferClIterator *i;
  cl_mem                image = NULL, blur = NULL;
  cl_int                cl_err = CL_SUCCESS;
  cl_int                stride, x0, y0, width, screen;
  cl_int                clip_low, clip_high;
  cl_float4             luminance;
  cl_float              in_offset, scale, out_offset, low_limit, high_limit;
  gfloat                in_range, out_range;
  gdouble               lum[3];
  gdouble               in_low, in_high, out_low, out_high;
  gboolean              err = FALSE;
  gboolean              ok  = FALSE;
  size_t                n, global_ws;

  screen     = o->limit_exposure;
  out_format = babl_format (screen ? "RaGaBaA float" : "RGBA float");

  /* The kernels compute the graph for sRGB RGB(A) pixels */
  if (gegl_operation_get_format (operation, "output") != out_format ||
      babl_format_get_space (gegl_buffer_get_format (input)) !=
      babl_space ("sRGB"))
    return FALSE;

  if (gegl_rectangle_is_empty (result) ||
      ! gegl_rectangle_contains (extent, result))
    return FALSE;

  if (! cl_data)
    {
      const char *kernel_name[] = { "bloom_glow", "bloom_combine", NULL };

      cl_data = gegl_cl_compile_and_build (bloom_cl_source, kernel_name);
    }

  if (! cl_data)
    return FALSE;

  babl_space_get_rgb_luminance (babl_space ("sRGB"),
                                &lum[0], &lum[1], &lum[2]);
  luminance.s[0] = lum[0];
  luminance.s[1] = lum[1];
  luminance.s[2] = lum[2];
  luminance.s[3] = 0.0f;

  /* As gegl:levels computes them */
  in_low   = (o->threshold - o->softness) / 100.0;
  in_high  = (o->threshold + o->softness) / 100.0;
  out_low  = 0.0;
  out_high = o->strength / 100.0;

  in_offset  = in_low * 1.0;
  out_offset = out_low * 1.0;
  in_range   = in_high - in_low;
  out_range  = out_high - out_low;

  if (fabsf (in_range) <= 1e-6)
    in_range = copysignf (1e-6, in_range);

  scale = out_range / in_range;

  /* As gegl:rgb-clip with low-limit 0.0 does */
  clip_low   = TRUE;
  low_limit  = 0.0;
  clip_high  = TRUE;
  high_limit = o->strength / 100.0;

  n = (size_t) extent->width * extent->height;

  image = gegl_clCreateBuffer (gegl_cl_get_context (), CL_MEM_READ_WRITE,
                               n * 4 * sizeof (cl_float), NULL, &cl_err);
  if (cl_err != CL_SUCCESS)
    goto cleanup;

  blur = gegl_clCreateBuffer (gegl_cl_get_context (), CL_MEM_READ_WRITE,
                              n * 4 * sizeof (cl_float), NULL, &cl_err);
  if (cl_err != CL_SUCCESS)
    goto cleanup;

  if (! gblur_mem_cl_read (input, extent, rgba, image, FALSE))
    goto cleanup;

  /* the glow is computed in blur and then blurred in place */
  cl_err = gegl_cl_set_kernel_args (cl_data->kernel[0],
                                    sizeof (cl_mem),    &image,
                                    sizeof (cl_mem),    &blur,
                                    sizeof (cl_float4), &luminance,
                                    sizeof (cl_float),  &in_offset,
                                    sizeof (cl_float),  &scale,
                                    sizeof (cl_float),  &out_offset,
                                    sizeof (cl_int),    &clip_low,
                                    sizeof (cl_float),  &low_limit,
                                    sizeof (cl_int),    &clip_high,
                                    sizeof (cl_float),  &high_limit,
                                    NULL);
  if (cl_err != CL_SUCCESS)
    goto cleanup;

  cl_err = gegl_clEnqueueNDRangeKernel (gegl_cl_get_command_queue (),
                                        cl_data->kernel[0], 1,
                                        NULL, &n, NULL, 0, NULL, NULL);
  if (cl_err != CL_SUCCESS)
    goto cleanup;

  if (! gblur_mem_cl_blur (blur, blur, extent->width, extent->height, 4,
                           o->radius, TRUE))
    goto cleanup;

  stride = extent->width;

  i = gegl_buffer_cl_iterator_new (output, result, out_format,
                                   GEGL_CL_BUFFER_WRITE);

  while (gegl_buffer_cl_iterator_next (i, &err) && ! err)
    {
      const GeglRectangle *roi = &i->roi[0];

      x0        = roi->x - extent->x;
      y0        = roi->y - extent->y;
      width     = roi->width;
      global_ws = i->size[0];

      cl_err = gegl_cl_set_kernel_args (cl_data->kernel[1],
                                        sizeof (cl_mem), &image,
                                        sizeof (cl_mem), &blur,
                                        sizeof (cl_mem), &i->tex[0],
                                        sizeof (cl_int), &stride,
                                        sizeof (cl_int), &x0,
                                        sizeof (cl_int), &y0,
                                        sizeof (cl_int), &width,
                                        sizeof (cl_int), &screen,
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
  g_object_set (gegl, "use-opencl", gegl_operation_use_opencl (operation),
                NULL);

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
    "name",        "gegl:bloom",
    "title",       _("Bloom"),
    "categories",  "light",
    "reference-hash", "ab23acffc881bde3fa22458bba89e9ed",
    "description", _("Add glow around bright areas"),
    NULL);
}

#endif
