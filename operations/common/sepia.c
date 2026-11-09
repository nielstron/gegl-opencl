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
 * Copyright 2015 Red Hat, Inc.
 */

#include "config.h"
#include <glib/gi18n-lib.h>

#ifdef GEGL_PROPERTIES

property_double (scale, _("Effect strength"), 1.0)
    description(_("Strength of the sepia effect"))
    value_range (0.0, 1.0)

property_boolean (srgb, _("sRGB"), TRUE)
    description (_("Use sRGB gamma instead of linear"))

#else

#define GEGL_OP_POINT_FILTER
#define GEGL_OP_NAME     sepia
#define GEGL_OP_C_SOURCE sepia.c

#include "gegl-op.h"
#include "opencl/gegl-cl.h"
#include "opencl/sepia.cl.h"

static void prepare (GeglOperation *operation)
{
  const Babl *space = gegl_operation_get_source_space (operation, "input");
  GeglProperties *o = GEGL_PROPERTIES (operation);
  const Babl *format;

  if (o->srgb)
    format = babl_format_with_space ("R~G~B~A float", space);
  else
    format = babl_format_with_space ("RGBA float", space);

  gegl_operation_set_format (operation, "input", format);
  gegl_operation_set_format (operation, "output", format);
}

static void
get_matrix (gdouble scale,
            gfloat  m[3][4])
{
  gdouble remaining = 1.0 - scale;

  m[0][0] = 0.393 + 0.607 * remaining;
  m[0][1] = 0.769 - 0.769 * remaining;
  m[0][2] = 0.189 - 0.189 * remaining;
  m[0][3] = 0.0f;

  m[1][0] = 0.349 - 0.349 * remaining;
  m[1][1] = 0.686 + 0.314 * remaining;
  m[1][2] = 0.168 - 0.168 * remaining;
  m[1][3] = 0.0f;

  m[2][0] = 0.272 - 0.272 * remaining;
  m[2][1] = 0.534 - 0.534 * remaining;
  m[2][2] = 0.131 + 0.869 * remaining;
  m[2][3] = 0.0f;
}

static gboolean
process (GeglOperation       *operation,
         void                *in_buf,
         void                *out_buf,
         glong                n_pixels,
         const GeglRectangle *roi,
         gint                 level)
{
  GeglProperties *o = GEGL_PROPERTIES (operation);
  const gfloat *in = in_buf;
  gfloat *out = out_buf;
  gfloat m[3][4];

  get_matrix (o->scale, m);

  for (glong i = 0; i < n_pixels; i++)
    {
      gfloat r = in[0], g = in[1], b = in[2];

      for (gint c = 0; c < 3; c++)
        out[c] = m[c][0] * r + m[c][1] * g + m[c][2] * b;
      out[3] = in[3];

      in += 4;
      out += 4;
    }

  return TRUE;
}

static gboolean
cl_process (GeglOperation       *operation,
            cl_mem               in,
            cl_mem               out,
            size_t               n_pixels,
            const GeglRectangle *roi,
            gint                 level)
{
  static GeglClRunData *cl_data;
  const gchar *kernels[] = { "gegl_sepia", NULL };
  GeglProperties *o = GEGL_PROPERTIES (operation);
  gfloat m[3][4];
  cl_int cl_err;

  if (level != 0)
    return TRUE;

  if (!cl_data)
    cl_data = gegl_cl_compile_and_build (sepia_cl_source, kernels);
  if (!cl_data)
    return TRUE;

  get_matrix (o->scale, m);
  cl_err = gegl_cl_set_kernel_args (cl_data->kernel[0],
                                    sizeof (in), &in,
                                    sizeof (out), &out,
                                    sizeof (m[0]), m[0],
                                    sizeof (m[1]), m[1],
                                    sizeof (m[2]), m[2],
                                    NULL);
  CL_CHECK;
  cl_err = gegl_clEnqueueNDRangeKernel (gegl_cl_get_command_queue (),
                                       cl_data->kernel[0], 1,
                                       NULL, &n_pixels, NULL, 0, NULL, NULL);
  CL_CHECK;
  return FALSE;

error:
  return TRUE;
}

static void
gegl_op_class_init (GeglOpClass *klass)
{
  GeglOperationClass *operation_class = GEGL_OPERATION_CLASS (klass);
  GeglOperationPointFilterClass *point_filter_class =
    GEGL_OPERATION_POINT_FILTER_CLASS (klass);

  operation_class->prepare = prepare;
  operation_class->opencl_support = TRUE;

  point_filter_class->process = process;
  point_filter_class->cl_process = cl_process;

  gegl_operation_class_set_keys (operation_class,
    "name"       , "gegl:sepia",
    "title",       _("Sepia"),
    "reference-hash", "60d6e6b36e4305931dc5cdca8201404e",
    "categories" , "color",
    "description", _("Apply a sepia tone to the input image"),
    NULL);
}

#endif
