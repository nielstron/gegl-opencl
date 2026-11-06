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
 * Copyright 1995 Spencer Kimball and Peter Mattis
 * Copyright 1996 Torsten Martinsen
 * Copyright 2000 Tim Copperfield <timecop@japan.co.jp>
 * Copyright 2012 Maxime Nicco <maxime.nicco@gmail.com>
 * Copyright 2012,2016 Øyvind Kolås <pippin@gimp.org>
 */

#include "config.h"
#include <glib/gi18n-lib.h>

#ifdef GEGL_PROPERTIES

property_boolean (correlated, _("Correlated noise"), FALSE)

property_boolean (independent, _("Independent RGB"), TRUE)
   description (_("Control amount of noise for each RGB channel separately"))

property_boolean (linear, _("Linear RGB"), TRUE)
   description (_("Operate on linearized RGB color data"))

property_boolean (gaussian, _("Gaussian distribution"), TRUE)
   description (_("Use a gaussian noise distribution, when unticked a linear noise distribution is used instead"))

property_double (red, _("Red"), 0.20)
   value_range  (0.0, 1.0)
   ui_meta      ("label", "[! independent : non-independent-label]")
   ui_meta      ("non-independent-label", _("Value"))

property_double (green, _("Green"), 0.20)
   value_range  (0.0, 1.0)
   ui_meta      ("visible", "independent")

property_double (blue, _("Blue"), 0.20)
   value_range  (0.0, 1.0)
   ui_meta      ("visible", "independent")

property_double (alpha, _("Alpha"), 0.0)
   value_range  (0.0, 1.0)

property_seed (seed, _("Random seed"), rand)

#else

#define GEGL_OP_POINT_FILTER
#define GEGL_OP_NAME     noise_rgb
#define GEGL_OP_C_SOURCE noise-rgb.c

#include "gegl-op.h"
#include <stdio.h>
#include <stdlib.h>

/*
 * Return a Gaussian (aka normal) distributed random variable.
 *
 * Adapted from gauss.c included in GNU scientific library.
 *
 * Ratio method (Kinderman-Monahan); see Knuth v2, 3rd ed, p130
 * K+M, ACM Trans Math Software 3 (1977) 257-260.
*/
static gfloat
noise_gauss (GeglRandom *rand, gint xx, gint yy, gint *n)
{
  gfloat u, v, x;

  do
  {
    v = gegl_random_float (rand, xx, yy, 0, (*n)++);

    do
      u = gegl_random_float (rand, xx, yy, 0, (*n)++);
    while (u == 0);

    /* Const 1.715... = sqrt(8/e) */
    x = 1.71552776992141359295 * (v - 0.5f) / u;
  }
  while (x * x > -4.0f * logf (u));

  return x;
}

static gfloat
noise_linear (GeglRandom *rand, gint xx, gint yy, gint *n)
{
  return gegl_random_float (rand, xx, yy, 0, (*n)++) * 2 - 1.0;
}

static void
prepare (GeglOperation *operation)
{
  const Babl *space = gegl_operation_get_source_space (operation, "input");
  GeglProperties *o  = GEGL_PROPERTIES (operation);
  if (o->linear)
    {
      gegl_operation_set_format (operation, "input",
                                 babl_format_with_space ("RGBA float", space));
      gegl_operation_set_format (operation, "output",
                                 babl_format_with_space ("RGBA float", space));
    }
  else
    {
      gegl_operation_set_format (operation, "input",
                                 babl_format_with_space ("R~G~B~A float", space));
      gegl_operation_set_format (operation, "output",
                                 babl_format_with_space ("R~G~B~A float", space));
    }
}

static gboolean
process (GeglOperation       *operation,
         void                *in_buf,
         void                *out_buf,
         glong                n_pixels,
         const GeglRectangle *roi,
         gint                 level)
{
  GeglProperties *o  = GEGL_PROPERTIES (operation);

  gdouble  noise_coeff = 0.0;
  gint     b, i;
  gint     x, y;
  gdouble  noise[4];
  gfloat   tmp;
  gfloat   * GEGL_ALIGNED in_pixel;
  gfloat   * GEGL_ALIGNED out_pixel;
  gfloat   (*noise_fun) (GeglRandom *rand, gint xx, gint yy, gint *n) = noise_gauss;

  in_pixel   = in_buf;
  out_pixel  = out_buf;

  noise[0] = o->red;
  noise[1] = o->green;
  noise[2] = o->blue;
  noise[3] = o->alpha;

  if (o->gaussian == FALSE)
    noise_fun = noise_linear;

  x = roi->x;
  y = roi->y;

  for (i=0; i<n_pixels; i++)
  {
    gint n = 0;

    for (b = 0; b < 4; b++)
    {
      if (b == 0 || o->independent || b == 3 )
         noise_coeff = noise[b] * noise_fun (o->rand, x, y, &n) * 0.5;

      if (noise_coeff != 0.0)
      {
        if (o->correlated)
        {
          tmp = (in_pixel[b] + (in_pixel[b] * (noise_coeff / 0.5)) );
        }
        else
        {
          tmp = (in_pixel[b] + noise_coeff );
        }

        out_pixel[b] = CLAMP(tmp, 0.0, 1.0);
      }
      else
      {
        out_pixel[b] = in_pixel[b];
      }
    }

    in_pixel  += 4;
    out_pixel += 4;

    x++;
    if (x >= roi->x + roi->width)
      {
        x = roi->x;
        y++;
      }
  }

  return TRUE;
}

#include "opencl/gegl-cl.h"
#include "opencl/noise-rgb.cl.h"

static GeglClRunData *cl_data   = NULL;
static cl_mem         log_table = NULL;

/* -4.0f * logf (u) for every u gegl_random_float () can return, so that
 * the kernel takes the same rejection decisions as noise_gauss ()
 */
static cl_mem
get_log_table (cl_int *cl_err)
{
  if (! log_table)
    {
      gfloat *table = g_new (gfloat, 65536);
      gint    k;

      table[0] = 0.0f; /* u == 0 is never used */

      for (k = 1; k < 65536; k++)
        {
          gfloat u = k * 0.00001525902189669642175;

          table[k] = -4.0f * logf (u);
        }

      log_table = gegl_clCreateBuffer (gegl_cl_get_context (),
                                       CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY,
                                       65536 * sizeof (gfloat),
                                       table, cl_err);
      g_free (table);

      if (*cl_err != CL_SUCCESS)
        log_table = NULL;
    }
  else
    {
      *cl_err = CL_SUCCESS;
    }

  return log_table;
}

static gboolean
cl_process (GeglOperation       *operation,
            cl_mem               in,
            cl_mem               out,
            size_t               global_worksize,
            const GeglRectangle *roi,
            gint                 level)
{
  GeglProperties *o           = GEGL_PROPERTIES (operation);
  cl_int          cl_err      = 0;
  cl_mem          cl_random_data;
  cl_mem          cl_log_table;
  cl_ushort4      rand;
  cl_float4       amount;
  cl_int          correlated  = o->correlated;
  cl_int          independent = o->independent;
  cl_int          gaussian    = o->gaussian;

  if (!cl_data)
    {
      const char *kernel_name[] = { "cl_noise_rgb", NULL };
      cl_data = gegl_cl_compile_and_build (noise_rgb_cl_source, kernel_name);
    }

  if (!cl_data)
    return TRUE;

  gegl_cl_random_get_ushort4 (o->rand, &rand);

  cl_random_data = gegl_cl_load_random_data (&cl_err);
  CL_CHECK;

  cl_log_table = get_log_table (&cl_err);
  CL_CHECK;

  amount.s[0] = o->red;
  amount.s[1] = o->green;
  amount.s[2] = o->blue;
  amount.s[3] = o->alpha;

  cl_err = gegl_cl_set_kernel_args (cl_data->kernel[0],
                                    sizeof (cl_mem),     &in,
                                    sizeof (cl_mem),     &out,
                                    sizeof (cl_mem),     &cl_random_data,
                                    sizeof (cl_mem),     &cl_log_table,
                                    sizeof (cl_ushort4), &rand,
                                    sizeof (cl_int),     &roi->x,
                                    sizeof (cl_int),     &roi->y,
                                    sizeof (cl_int),     &roi->width,
                                    sizeof (cl_float4),  &amount,
                                    sizeof (cl_int),     &correlated,
                                    sizeof (cl_int),     &independent,
                                    sizeof (cl_int),     &gaussian,
                                    NULL);
  CL_CHECK;

  cl_err = gegl_clEnqueueNDRangeKernel (gegl_cl_get_command_queue (),
                                        cl_data->kernel[0], 1,
                                        NULL, &global_worksize, NULL,
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

  operation_class->prepare        = prepare;
  operation_class->opencl_support = TRUE;
  point_filter_class->process     = process;
  point_filter_class->cl_process  = cl_process;

  gegl_operation_class_set_keys (operation_class,
    "name",               "gegl:noise-rgb",
    "title",              _("Add RGB Noise"),
    "categories",         "noise",
    "position-dependent", "true",
    "reference-hash",     "b92f11f3a5f288c4f0e20afb0bba054f",
    "description", _("Distort colors by random amounts"),
    NULL);
}

#endif
