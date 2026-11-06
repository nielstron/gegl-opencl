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
 * Copyright 1996 Federico Mena Quintero
 * Copyright 1997 Scott Goehring
 * Copyright 2006 Øyvind Kolås <pippin@gimp.org>
 * Copyright 2013 Carlos Zubieta <czubieta.dev@gmail.com>
 *
 */

#include "config.h"
#include <glib/gi18n-lib.h>

#ifdef GEGL_PROPERTIES

#else

#define GEGL_OP_FILTER
#define GEGL_OP_NAME     stretch_contrast_hsv
#define GEGL_OP_C_SOURCE stretch-contrast-hsv.c

#include "gegl-op.h"

typedef struct {
  gfloat slo;
  gfloat sdiff;
  gfloat vlo;
  gfloat vdiff;
} AutostretchData;

static void
buffer_get_auto_stretch_data (GeglOperation       *operation,
                              GeglBuffer          *buffer,
                              const GeglRectangle *result,
                              AutostretchData     *data,
                              const Babl          *space)
{
  gfloat smin =  G_MAXFLOAT;
  gfloat smax = -G_MAXFLOAT;
  gfloat vmin =  G_MAXFLOAT;
  gfloat vmax = -G_MAXFLOAT;

  GeglBufferIterator *gi;
  gint                done_pixels = 0;

  gegl_operation_progress (operation, 0.0, "");

  gi = gegl_buffer_iterator_new (buffer, result, 0, babl_format_with_space ("HSVA float", space),
                                 GEGL_ACCESS_READ, GEGL_ABYSS_NONE, 1);

  while (gegl_buffer_iterator_next (gi))
    {
      gfloat *buf = gi->items[0].data;
      gint    i;

      for (i = 0; i < gi->length; i++)
        {
          gfloat sval = buf[1];
          gfloat vval = buf[2];

          smin = MIN (sval, smin);
          smax = MAX (sval, smax);
          vmin = MIN (vval, vmin);
          vmax = MAX (vval, vmax);

          buf += 4;
        }

      done_pixels += gi->length;

      gegl_operation_progress (operation,
                               (gdouble) 0.5 * done_pixels /
                               (gdouble) (result->width * result->height),
                               "");
    }

  if (data)
    {
      data->slo   = smin;
      data->sdiff = smax - smin;
      data->vlo   = vmin;
      data->vdiff = vmax - vmin;
    }

  gegl_operation_progress (operation, 0.5, "");
}

static void
clean_autostretch_data (AutostretchData *data)
{
  if (data->sdiff < GEGL_FLOAT_EPSILON)
    {
      data->sdiff = 1.0;
      data->slo   = 0.0;
    }

  if (data->vdiff < GEGL_FLOAT_EPSILON)
    {
      data->vdiff = 1.0;
      data->vlo   = 0.0;
    }
}

static void
prepare (GeglOperation *operation)
{
  const Babl *space = gegl_operation_get_source_space (operation, "input");
  const Babl *format;

  /* The OpenCL path converts to and from HSV itself, so let it operate on
   * linear RGBA, avoiding two costly HSV conversions on the CPU.  babl's
   * HSV model is defined on the linear RGB of the space, encoded with the
   * sRGB TRC regardless of the space's own TRC, which the kernels apply.
   */
  if (gegl_operation_use_opencl (operation) &&
      (! space || babl_space_is_rgb (space)))
    format = babl_format_with_space ("RGBA float", space);
  else
    format = babl_format_with_space ("HSVA float", space);

  gegl_operation_set_format (operation, "input",  format);
  gegl_operation_set_format (operation, "output", format);
}

static GeglRectangle
get_required_for_output (GeglOperation       *operation,
                         const gchar         *input_pad,
                         const GeglRectangle *roi)
{
  GeglRectangle result = *gegl_operation_source_get_bounding_box (operation, "input");

  /* Don't request an infinite plane */
  if (gegl_rectangle_is_infinite_plane (&result))
    return *roi;

  return result;
}

static GeglRectangle
get_cached_region (GeglOperation       *operation,
                   const GeglRectangle *roi)
{
  GeglRectangle result = *gegl_operation_source_get_bounding_box (operation, "input");

  if (gegl_rectangle_is_infinite_plane (&result))
    return *roi;

  return result;
}

#include "opencl/gegl-cl.h"
#include "gegl-buffer-cl-iterator.h"
#include "opencl/stretch-contrast-hsv.cl.h"

static GeglClRunData *cl_data = NULL;

static gboolean
cl_build_kernels (void)
{
  if (!cl_data)
    {
      const char *kernel_name[] = {"hsv_sv_min_max",
                                   "stretch_contrast_hsv",
                                   NULL};
      cl_data = gegl_cl_compile_and_build (stretch_contrast_hsv_cl_source,
                                           kernel_name);
    }

  return cl_data == NULL;
}

/* Reduces in_tex to (min s, min v, -max s, -max v) and folds the result
 * into acc.
 */
static gboolean
cl_get_sv_min_max (cl_mem  in_tex,
                   size_t  n_pixels,
                   gfloat  acc[4])
{
  cl_int     cl_err   = 0;
  cl_mem     cl_part  = NULL;
  cl_float4  part[256];
  cl_int     n        = n_pixels;
  size_t     local_ws = 256;
  size_t     max_local_ws;
  size_t     work_groups;
  size_t     global_ws;
  gint       i;

  if (n_pixels < 1)
    return FALSE;

  cl_err = gegl_clGetDeviceInfo (gegl_cl_get_device (),
                                 CL_DEVICE_MAX_WORK_GROUP_SIZE,
                                 sizeof (size_t), &max_local_ws, NULL);
  CL_CHECK;

  max_local_ws = MIN (max_local_ws, cl_data->work_group_size[0]);

  /* Needs to be a power of two */
  while (local_ws > max_local_ws)
    local_ws /= 2;

  work_groups = MIN ((n_pixels + local_ws - 1) / local_ws, 256);
  global_ws   = work_groups * local_ws;

  cl_part = gegl_clCreateBuffer (gegl_cl_get_context (),
                                 CL_MEM_WRITE_ONLY,
                                 work_groups * sizeof (cl_float4),
                                 NULL, &cl_err);
  CL_CHECK;

  cl_err = gegl_cl_set_kernel_args (cl_data->kernel[0],
                                    sizeof (cl_mem), &in_tex,
                                    sizeof (cl_mem), &cl_part,
                                    local_ws * sizeof (cl_float4), NULL,
                                    sizeof (cl_int), &n,
                                    NULL);
  CL_CHECK;

  cl_err = gegl_clEnqueueNDRangeKernel (gegl_cl_get_command_queue (),
                                        cl_data->kernel[0], 1,
                                        NULL, &global_ws, &local_ws,
                                        0, NULL, NULL);
  CL_CHECK;

  cl_err = gegl_clEnqueueReadBuffer (gegl_cl_get_command_queue (),
                                     cl_part, CL_TRUE, 0,
                                     work_groups * sizeof (cl_float4), part,
                                     0, NULL, NULL);
  CL_CHECK;

  for (i = 0; i < work_groups; i++)
    {
      acc[0] = MIN (acc[0], part[i].s[0]);
      acc[1] = MIN (acc[1], part[i].s[1]);
      acc[2] = MIN (acc[2], part[i].s[2]);
      acc[3] = MIN (acc[3], part[i].s[3]);
    }

  cl_err = gegl_clReleaseMemObject (cl_part);
  CL_CHECK_ONLY (cl_err);

  return FALSE;

error:
  if (cl_part)
    gegl_clReleaseMemObject (cl_part);

  return TRUE;
}

static gboolean
cl_process (GeglOperation       *operation,
            GeglBuffer          *input,
            GeglBuffer          *output,
            const GeglRectangle *result)
{
  const Babl           *format = gegl_operation_get_format (operation, "output");
  GeglBufferClIterator *i;
  AutostretchData       data;
  gfloat                acc[4] = { G_MAXFLOAT, G_MAXFLOAT,
                                   G_MAXFLOAT, G_MAXFLOAT };
  cl_int                cl_err = 0;
  gboolean              err    = FALSE;
  gint                  read;

  /* The kernels expect linear RGBA, which prepare () only selects when
   * OpenCL is enabled.
   */
  if (!babl_model_is (babl_format_get_model (format), "RGBA"))
    return FALSE;

  if (cl_build_kernels ())
    return FALSE;

  i = gegl_buffer_cl_iterator_new (input, result, format,
                                   GEGL_CL_BUFFER_READ);

  while (gegl_buffer_cl_iterator_next (i, &err) && !err)
    {
      err = cl_get_sv_min_max (i->tex[0], i->size[0], acc);

      if (err)
        {
          gegl_buffer_cl_iterator_stop (i);
          break;
        }
    }

  if (err)
    return FALSE;

  data.slo   =  acc[0];
  data.sdiff = -acc[2] - acc[0];
  data.vlo   =  acc[1];
  data.vdiff = -acc[3] - acc[1];

  clean_autostretch_data (&data);

  i = gegl_buffer_cl_iterator_new (output, result, format,
                                   GEGL_CL_BUFFER_WRITE);

  read = gegl_buffer_cl_iterator_add (i, input, result, format,
                                      GEGL_CL_BUFFER_READ, GEGL_ABYSS_NONE);

  while (gegl_buffer_cl_iterator_next (i, &err) && !err)
    {
      cl_err = gegl_cl_set_kernel_args (cl_data->kernel[1],
                                        sizeof (cl_mem),   &i->tex[read],
                                        sizeof (cl_mem),   &i->tex[0],
                                        sizeof (cl_float), &data.slo,
                                        sizeof (cl_float), &data.sdiff,
                                        sizeof (cl_float), &data.vlo,
                                        sizeof (cl_float), &data.vdiff,
                                        NULL);
      CL_CHECK;

      cl_err = gegl_clEnqueueNDRangeKernel (gegl_cl_get_command_queue (),
                                            cl_data->kernel[1], 1,
                                            NULL, &i->size[0], NULL,
                                            0, NULL, NULL);
      CL_CHECK;
    }

  if (err)
    return FALSE;

  gegl_operation_progress (operation, 1.0, "");

  return TRUE;

error:
  gegl_buffer_cl_iterator_stop (i);

  return FALSE;
}

static gboolean
process (GeglOperation       *operation,
         GeglBuffer          *input,
         GeglBuffer          *output,
         const GeglRectangle *result,
         gint                 level)
{
  const Babl *space = gegl_operation_get_format (operation, "output");
  AutostretchData     data;
  GeglBufferIterator *gi;
  gint                done_pixels = 0;

  if (gegl_operation_use_opencl (operation))
    if (cl_process (operation, input, output, result))
      return TRUE;

  buffer_get_auto_stretch_data (operation, input, result, &data, space);
  clean_autostretch_data (&data);

  gegl_operation_progress (operation, 0.5, "");

  gi = gegl_buffer_iterator_new (input, result, 0, babl_format_with_space ("HSVA float", space),
                                 GEGL_ACCESS_READ, GEGL_ABYSS_NONE, 2);

  gegl_buffer_iterator_add (gi, output, result, 0, babl_format_with_space ("HSVA float", space),
                            GEGL_ACCESS_WRITE, GEGL_ABYSS_NONE);

  while (gegl_buffer_iterator_next (gi))
    {
      gfloat *in  = gi->items[0].data;
      gfloat *out = gi->items[1].data;
      gint    i;

      for (i = 0; i < gi->length; i++)
        {
          out[0] = in[0]; /* Keep hue */
          out[1] = (in[1] - data.slo) / data.sdiff;
          out[2] = (in[2] - data.vlo) / data.vdiff;
          out[3] = in[3]; /* Keep alpha */

          in  += 4;
          out += 4;
        }

      done_pixels += gi->length;

      gegl_operation_progress (operation,
                               0.5 +
                               (gdouble) 0.5 * done_pixels /
                               (gdouble) (result->width * result->height),
                               "");
    }

  gegl_operation_progress (operation, 1.0, "");

  return TRUE;
}

/* Pass-through when trying to perform a reduction on an infinite plane
 */
static gboolean
operation_process (GeglOperation        *operation,
                   GeglOperationContext *context,
                   const gchar          *output_prop,
                   const GeglRectangle  *result,
                   gint                  level)
{
  GeglOperationClass  *operation_class;

  const GeglRectangle *in_rect =
    gegl_operation_source_get_bounding_box (operation, "input");

  operation_class = GEGL_OPERATION_CLASS (gegl_op_parent_class);

  if (in_rect && gegl_rectangle_is_infinite_plane (in_rect))
    {
      gpointer in = gegl_operation_context_get_object (context, "input");
      gegl_operation_context_take_object (context, "output",
                                          g_object_ref (G_OBJECT (in)));
      return TRUE;
    }

  /* chain up, which will create the needed buffers for our actual
   * process function
   */
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

  filter_class->process                    = process;
  operation_class->prepare                 = prepare;
  operation_class->process                 = operation_process;
  operation_class->threaded                = FALSE;
  operation_class->get_required_for_output = get_required_for_output;
  operation_class->get_cached_region       = get_cached_region;
  operation_class->opencl_support          = TRUE;

  gegl_operation_class_set_keys (operation_class,
    "name",           "gegl:stretch-contrast-hsv",
    "title",          _("Stretch Contrast HSV"),
    "categories",     "color:enhance",
    "reference-hash", "c7802207f601127c78bf11314af1fc16",
    "description",
        _("Scales the components of the buffer to be in the 0.0-1.0 range. "
          "This improves images that make poor use of the available contrast "
          "(little contrast, very dark, or very bright images). "
          "This version differs from Contrast Autostretch in that it works "
          "in HSV space, and preserves hue."),
        NULL);
}

#endif
