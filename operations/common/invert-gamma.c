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
 * Copyright 2006 Øyvind Kolås <pippin@gimp.org>
 */

#include "config.h"
#include <glib/gi18n-lib.h>


#ifdef GEGL_PROPERTIES

   /* no properties */

#else

#define GEGL_OP_POINT_FILTER
#define GEGL_OP_NAME     invert_gamma
#define GEGL_OP_C_SOURCE invert-gamma.c

#include "gegl-op.h"

#define INVERT_GAMMA "~"
#include "invert-common.h"

#include "opencl/gegl-cl.h"
#include "opencl/invert-gamma.cl.h"
#include "opencl/invert-linear.cl.h"

static void
prepare_gamma (GeglOperation *operation)
{
  GeglProperties *o = GEGL_PROPERTIES (operation);
  const Babl *format = gegl_operation_get_source_format (operation, "input");

  /* The nonlinear and perceptual curves agree in the default sRGB space. */
  if (format == babl_format ("R'G'B'A u8"))
    o->user_data = process_rgba_u8;
  else if (format == babl_format ("R'G'B' u8"))
    o->user_data = process_rgb_u8;
  else
    {
      prepare (operation);
      return;
    }

  gegl_operation_set_format (operation, "input", format);
  gegl_operation_set_format (operation, "output", format);
}

static gboolean
cl_process (GeglOperation       *operation,
            cl_mem               in,
            cl_mem               out,
            size_t               n_pixels,
            const GeglRectangle *roi,
            gint                 level)
{
  static GeglClRunData *cl_data_u8;
  static GeglClRunData *cl_data_float;
  GeglProperties *o = GEGL_PROPERTIES (operation);
  ProcessFunc func = o->user_data;
  cl_kernel kernel;
  cl_int cl_err;

  if (level != 0)
    return TRUE;

  if (func == process_rgba_u8 || func == process_rgb_u8)
    {
      const gchar *kernels[] = { "gegl_invert_gamma_rgba_u8",
                                 "gegl_invert_gamma_rgb_u8", NULL };

      if (!cl_data_u8)
        cl_data_u8 = gegl_cl_compile_and_build (invert_gamma_cl_source, kernels);
      if (!cl_data_u8)
        return TRUE;

      kernel = cl_data_u8->kernel[func == process_rgb_u8];
    }
  else
    {
      const gchar *kernels[] = { "gegl_invert_linear", NULL };

      if (func != process_rgba_float)
        return TRUE;

      if (!cl_data_float)
        cl_data_float = gegl_cl_compile_and_build (invert_linear_cl_source, kernels);
      if (!cl_data_float)
        return TRUE;

      kernel = cl_data_float->kernel[0];
    }

  cl_err = gegl_cl_set_kernel_args (kernel,
                                    sizeof (in), &in,
                                    sizeof (out), &out,
                                    NULL);
  CL_CHECK;
  cl_err = gegl_clEnqueueNDRangeKernel (gegl_cl_get_command_queue (),
                                       kernel, 1, NULL, &n_pixels, NULL,
                                       0, NULL, NULL);
  CL_CHECK;
  return FALSE;

error:
  return TRUE;
}

static void
gegl_op_class_init (GeglOpClass *klass)
{
  GeglOperationClass            *operation_class;
  GeglOperationPointFilterClass *point_filter_class;

  operation_class    = GEGL_OPERATION_CLASS (klass);
  point_filter_class = GEGL_OPERATION_POINT_FILTER_CLASS (klass);

  operation_class->prepare     = prepare_gamma;
  operation_class->opencl_support = TRUE;
  point_filter_class->process  = process;
  point_filter_class->cl_process = cl_process;

  gegl_operation_class_set_keys (operation_class,
    "name"       , "gegl:invert-gamma",
    "title",      _("Invert in Perceptual space"),
    "categories" , "color",
    "reference-hash", "db07b9d85f2786db29560bd50ae0e7a1",
    "description",
       _("Invert the components (except alpha) perceptually, "
         "the result is the corresponding \"negative\" image."),
    NULL);
}

#endif
