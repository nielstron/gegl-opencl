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
 * Copyright 1997 Miles O'Neal <meo@rru.com>  http://www.rru.com/~meo/
 * Copyright 2012 Maxime Nicco <maxime.nicco@gmail.com>
 * Copyright 2013 Téo Mazars   <teo.mazars@ensimag.fr>
 */

/*
 *  PICK Operation
 *     We pick a pixel at random from the neighborhood of the current pixel.
 */

#include "config.h"

#include <glib/gi18n-lib.h>

#ifdef GEGL_PROPERTIES

property_double (pct_random, _("Randomization (%)"), 50.0)
    value_range (0.0, 100.0)

property_int    (repeat, _("Repeat"), 1)
    value_range (1, 100)

property_seed (seed, _("Random seed"), rand)

#else

#define GEGL_OP_AREA_FILTER
#define GEGL_OP_NAME     noise_pick
#define GEGL_OP_C_SOURCE noise-pick.c

#include "gegl-op.h"

static void
prepare (GeglOperation *operation)
{
  GeglOperationAreaFilter *op_area = GEGL_OPERATION_AREA_FILTER (operation);
  GeglProperties          *o       = GEGL_PROPERTIES (operation);
  const Babl              *format;

  op_area->left   =
  op_area->right  =
  op_area->top    =
  op_area->bottom = o->repeat;

  format = gegl_operation_get_source_format (operation, "input");

  gegl_operation_set_format (operation, "input", format);
  gegl_operation_set_format (operation, "output", format);
}


/* We avoid unecessary calls to gegl_random
 * by recomputing a small part ourselves.
 * see gegl/gegl-random.c for details */
#define RAND_UINT_TO_FLOAT(x) (((x) & 0xffff) * 0.00001525902189669642175)

#include "opencl/gegl-cl.h"
#include "gegl-buffer-cl-iterator.h"
#include "opencl/noise-pick.cl.h"

static GeglClRunData *cl_data = NULL;

/* largest (rand & 0xffff) for which the CPU path moves the pixel, -1 if
 * there is none; computed with the same expression as the CPU path */
static gint
compute_max_k (gdouble pct_random)
{
  gint k;

  for (k = 0xffff; k >= 0; k--)
    {
      gfloat pct = RAND_UINT_TO_FLOAT ((guint) k) * 100.0;

      if (pct <= pct_random)
        break;
    }

  return k;
}

static gboolean
cl_noise_pick (cl_mem               in_tex,
               cl_mem               out_tex,
               const GeglRectangle *roi,
               gint                 in_width,
               gint                 area,
               gint                 repeat,
               gint                 max_k,
               gint                 bpp,
               GeglRandom          *rand)
{
  cl_int     cl_err = 0;
  cl_mem     cl_random_data;
  cl_ushort4 cl_rand;
  size_t     global_ws = (size_t) roi->width * roi->height;

  if (!cl_data)
    {
      const char *kernel_name[] = { "cl_noise_pick", NULL };
      cl_data = gegl_cl_compile_and_build (noise_pick_cl_source, kernel_name);
    }

  if (!cl_data)
    return TRUE;

  gegl_cl_random_get_ushort4 (rand, &cl_rand);

  cl_random_data = gegl_cl_load_random_data (&cl_err);
  CL_CHECK;

  cl_err = gegl_cl_set_kernel_args (cl_data->kernel[0],
                                    sizeof (cl_mem),     &in_tex,
                                    sizeof (cl_mem),     &out_tex,
                                    sizeof (cl_mem),     &cl_random_data,
                                    sizeof (cl_ushort4), &cl_rand,
                                    sizeof (cl_int),     &roi->x,
                                    sizeof (cl_int),     &roi->y,
                                    sizeof (cl_int),     &roi->width,
                                    sizeof (cl_int),     &in_width,
                                    sizeof (cl_int),     &area,
                                    sizeof (cl_int),     &repeat,
                                    sizeof (cl_int),     &max_k,
                                    sizeof (cl_int),     &bpp,
                                    NULL);
  CL_CHECK;

  cl_err = gegl_clEnqueueNDRangeKernel (gegl_cl_get_command_queue (),
                                        cl_data->kernel[0], 1,
                                        NULL, &global_ws, NULL,
                                        0, NULL, NULL);
  CL_CHECK;

  return FALSE;

error:
  return TRUE;
}

static gboolean
cl_process (GeglOperation       *operation,
            GeglBuffer          *input,
            GeglBuffer          *output,
            const GeglRectangle *result,
            const Babl          *format)
{
  GeglOperationAreaFilter *op_area = GEGL_OPERATION_AREA_FILTER (operation);
  GeglProperties          *o       = GEGL_PROPERTIES (operation);
  gint                     bpp     = babl_format_get_bytes_per_pixel (format);
  gint                     max_k   = compute_max_k (o->pct_random);
  GeglBufferClIterator    *i;
  gint                     read;
  gint                     err = 0;

  i = gegl_buffer_cl_iterator_new (output, result, format,
                                   GEGL_CL_BUFFER_WRITE);

  read = gegl_buffer_cl_iterator_add_2 (i, input, result, format,
                                        GEGL_CL_BUFFER_READ,
                                        op_area->left, op_area->right,
                                        op_area->top,  op_area->bottom,
                                        GEGL_ABYSS_CLAMP);

  while (gegl_buffer_cl_iterator_next (i, &err) && !err)
    {
      err = cl_noise_pick (i->tex[read], i->tex[0], &i->roi[0],
                           i->roi[read].width, op_area->left,
                           o->repeat, max_k, bpp, o->rand);

      if (err)
        {
          gegl_buffer_cl_iterator_stop (i);
          break;
        }
    }

  return !err;
}

static gboolean
process (GeglOperation       *operation,
         GeglBuffer          *input,
         GeglBuffer          *output,
         const GeglRectangle *result,
         gint                 level)
{
  GeglProperties     *o;
  const Babl         *format;
  gint                bpp;
  GeglBufferIterator *gi;
  GeglSampler        *sampler;

  o = GEGL_PROPERTIES (operation);

  format = gegl_operation_get_source_format (operation, "input");
  bpp = babl_format_get_bytes_per_pixel (format);

  if (level == 0 && gegl_operation_use_opencl (operation))
    if (cl_process (operation, input, output, result, format))
      return TRUE;

  gi = gegl_buffer_iterator_new (output, result, 0, format,
                                 GEGL_ACCESS_WRITE, GEGL_ABYSS_CLAMP, 1);

  sampler = gegl_buffer_sampler_new_at_level (input, format, GEGL_SAMPLER_NEAREST, level);

  while (gegl_buffer_iterator_next (gi))
    {
      gchar        *data = gi->items[0].data;
      GeglRectangle roi  = gi->items[0].roi;
      gint          i, j;

      for (j = roi.y; j < roi.y + roi.height ; j++)
        for (i = roi.x; i < roi.x + roi.width ; i++)
          {
            gint r;
            gint pos_x = i, pos_y = j;

            for (r = 0; r < o->repeat; r++)
              {
                guint  rand = gegl_random_int (o->rand, pos_x, pos_y, 0, r);
                gfloat pct  = RAND_UINT_TO_FLOAT (rand) * 100.0;

                if (pct <= o->pct_random)
                  {
                    gint rand2 = (gint) (rand % 9);

                    pos_x += (rand2 % 3) - 1;
                    pos_y += (rand2 / 3) - 1;
                  }
              }

            gegl_sampler_get (sampler, pos_x, pos_y, NULL, data, GEGL_ABYSS_CLAMP);
            data += bpp;
          }
    }
  g_object_unref (sampler);

  return TRUE;
}

static void
gegl_op_class_init (GeglOpClass *klass)
{
  GeglOperationClass       *operation_class;
  GeglOperationFilterClass *filter_class;

  operation_class = GEGL_OPERATION_CLASS (klass);
  filter_class    = GEGL_OPERATION_FILTER_CLASS (klass);

  operation_class->prepare        = prepare;
  operation_class->opencl_support = TRUE;
  filter_class->process           = process;

  gegl_operation_class_set_keys (operation_class,
    "name",       "gegl:noise-pick",
    "title",      _("Noise Pick"),
    "categories", "noise",
    "reference-hash", "0634d248731c51f4784749b934c05bd2",
    "description", _("Randomly interchange some pixels with neighbors"),
    NULL);
}

#endif
