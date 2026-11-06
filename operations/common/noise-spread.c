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
 * Copyright (C) 2011 Barak Itkin <lightningismyname@gmail.org>
 *
 * Based on "Spread" (Noise) GIMP plugin
 * Copyright (C) 1997 Brian Degenhardt and Federico Mena Quintero
 *
 * The workshop/whirl-pinch.c and common/pixelise.c were used as
 * templates for this op file.
 */
#include "config.h"
#include <glib/gi18n-lib.h>

#ifdef GEGL_PROPERTIES

property_int    (amount_x, _("Horizontal"), 5)
    description (_("Horizontal spread amount"))
    value_range (0, 512)
    ui_meta     ("unit", "pixel-distance")
    ui_meta     ("axis", "x")

property_int    (amount_y, _("Vertical"), 5)
    description (_("Vertical spread amount"))
    value_range (0, 512)
    ui_meta     ("unit", "pixel-distance")
    ui_meta     ("axis", "y")

property_seed (seed, _("Random seed"), rand)

#else

#define GEGL_OP_AREA_FILTER
#define GEGL_OP_NAME     noise_spread
#define GEGL_OP_C_SOURCE noise-spread.c

#include "gegl-op.h"

static inline void
calc_sample_coords (gint        src_x,
                    gint        src_y,
                    gint        amount_x,
                    gint        amount_y,
                    GeglRandom *rand,
                    gint       *x,
                    gint       *y)
{
  gdouble angle;
  gint    xdist, ydist;

  /* get random angle, x distance, and y distance */
  xdist = amount_x > 0 ? gegl_random_int_range (rand, src_x, src_y, 0, 0,
                                                -amount_x, amount_x + 1) : 0;
  ydist = amount_y > 0 ? gegl_random_int_range (rand, src_x, src_y, 0, 1,
                                                -amount_y, amount_y + 1) : 0;
  angle = gegl_random_float_range (rand, src_x, src_y, 0, 2, -G_PI, G_PI);

  *x = src_x + floor (sin (angle) * xdist);
  *y = src_y + floor (cos (angle) * ydist);
}


static void
prepare (GeglOperation *operation)
{
  GeglProperties          *o       = GEGL_PROPERTIES (operation);
  GeglOperationAreaFilter *op_area = GEGL_OPERATION_AREA_FILTER (operation);
  const Babl              *format;

  op_area->left   =
  op_area->right  = (o->amount_x + 1) / 2;
  op_area->top    =
  op_area->bottom = (o->amount_y + 1) / 2;

  format = gegl_operation_get_source_format (operation, "input");

  gegl_operation_set_format (operation, "input", format);
  gegl_operation_set_format (operation, "output", format);
}

#include "opencl/gegl-cl.h"
#include "gegl-buffer-cl-iterator.h"
#include "opencl/noise-spread.cl.h"

static GeglClRunData *cl_data     = NULL;
static cl_mem         angle_table = NULL;

/* sin and cos of every angle gegl_random_float_range() can return in
 * calc_sample_coords(), indexed by the low 16 bits of the random number,
 * each split into a high and a low float so the kernel can reproduce the
 * double precision floor () of the CPU version.
 */
static cl_mem
get_angle_table (cl_int *cl_err)
{
  if (! angle_table)
    {
      const gfloat  min = -G_PI;
      const gfloat  max =  G_PI;
      cl_float4    *table;
      gint          k;

      table = g_new (cl_float4, 65536);

      for (k = 0; k < 65536; k++)
        {
          /* same expressions as gegl_random_float_range () */
          gfloat  f     = k * 0.00001525902189669642175;
          gfloat  angle = f * (max - min) + min;
          gdouble s     = sin (angle);
          gdouble c     = cos (angle);

          table[k].s[0] = s;
          table[k].s[1] = s - (gfloat) s;
          table[k].s[2] = c;
          table[k].s[3] = c - (gfloat) c;
        }

      angle_table = gegl_clCreateBuffer (gegl_cl_get_context (),
                                         CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY,
                                         65536 * sizeof (cl_float4),
                                         table, cl_err);
      g_free (table);

      if (*cl_err != CL_SUCCESS)
        angle_table = NULL;
    }
  else
    {
      *cl_err = CL_SUCCESS;
    }

  return angle_table;
}

static gboolean
cl_noise_spread (cl_mem               in_tex,
                 cl_mem               out_tex,
                 const GeglRectangle *roi,
                 gint                 in_width,
                 gint                 area_x,
                 gint                 area_y,
                 gint                 bpp,
                 GeglRandom          *rand)
{
  cl_int     cl_err = 0;
  cl_mem     cl_random_data;
  cl_mem     cl_angle_table;
  cl_ushort4 cl_rand;
  size_t     global_ws = (size_t) roi->width * roi->height;

  if (!cl_data)
    {
      const char *kernel_name[] = { "cl_noise_spread", NULL };
      cl_data = gegl_cl_compile_and_build (noise_spread_cl_source, kernel_name);
    }

  if (!cl_data)
    return TRUE;

  gegl_cl_random_get_ushort4 (rand, &cl_rand);

  cl_random_data = gegl_cl_load_random_data (&cl_err);
  CL_CHECK;

  cl_angle_table = get_angle_table (&cl_err);
  CL_CHECK;

  cl_err = gegl_cl_set_kernel_args (cl_data->kernel[0],
                                    sizeof (cl_mem),     &in_tex,
                                    sizeof (cl_mem),     &out_tex,
                                    sizeof (cl_mem),     &cl_random_data,
                                    sizeof (cl_mem),     &cl_angle_table,
                                    sizeof (cl_ushort4), &cl_rand,
                                    sizeof (cl_int),     &roi->x,
                                    sizeof (cl_int),     &roi->y,
                                    sizeof (cl_int),     &roi->width,
                                    sizeof (cl_int),     &in_width,
                                    sizeof (cl_int),     &area_x,
                                    sizeof (cl_int),     &area_y,
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
            const Babl          *format,
            gint                 amount_x,
            gint                 amount_y)
{
  GeglProperties       *o   = GEGL_PROPERTIES (operation);
  gint                  bpp = babl_format_get_bytes_per_pixel (format);
  GeglBufferClIterator *i;
  gint                  read;
  gint                  err = 0;

  i = gegl_buffer_cl_iterator_new (output, result, format,
                                   GEGL_CL_BUFFER_WRITE);

  read = gegl_buffer_cl_iterator_add_2 (i, input, result, format,
                                        GEGL_CL_BUFFER_READ,
                                        amount_x, amount_x,
                                        amount_y, amount_y,
                                        GEGL_ABYSS_CLAMP);

  while (gegl_buffer_cl_iterator_next (i, &err) && !err)
    {
      err = cl_noise_spread (i->tex[read], i->tex[0], &i->roi[0],
                             i->roi[read].width, amount_x, amount_y,
                             bpp, o->rand);

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
  GeglProperties         *o;
  const Babl         *format;
  gint                bpp;
  GeglBufferIterator *gi;
  gint                amount_x;
  gint                amount_y;
  GeglSampler        *sampler;

  o = GEGL_PROPERTIES (operation);

  amount_x = (o->amount_x + 1) / 2;
  amount_y = (o->amount_y + 1) / 2;

  format = gegl_operation_get_source_format (operation, "input");
  bpp = babl_format_get_bytes_per_pixel (format);

  if (level == 0 && gegl_operation_use_opencl (operation))
    if (cl_process (operation, input, output, result, format,
                    amount_x, amount_y))
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
            gint x, y;

            calc_sample_coords (i, j, amount_x, amount_y, o->rand, &x, &y);

            gegl_sampler_get (sampler, x, y, NULL, data, GEGL_ABYSS_CLAMP);
            data += bpp;
          }
    }

  g_object_unref (sampler);

  return TRUE;
}

static gboolean
operation_process (GeglOperation        *operation,
                   GeglOperationContext *context,
                   const gchar          *output_prop,
                   const GeglRectangle  *result,
                   gint                  level)
{
  GeglProperties      *o = GEGL_PROPERTIES (operation);
  GeglOperationClass  *operation_class;

  operation_class = GEGL_OPERATION_CLASS (gegl_op_parent_class);

  if (! o->amount_x && ! o->amount_y)
    {
      gpointer in = gegl_operation_context_get_object (context, "input");
      gegl_operation_context_take_object (context, "output",
                                          g_object_ref (G_OBJECT (in)));
      return TRUE;
    }

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

  operation_class->prepare        = prepare;
  operation_class->process        = operation_process;
  operation_class->opencl_support = TRUE;
  filter_class->process           = process;

  gegl_operation_class_set_keys (operation_class,
    "name",        "gegl:noise-spread",
    "title",       _("Noise Spread"),
    "categories",  "noise",
    "reference-hash", "c53483232973300e961be9819026f997",
    "description", _("Move pixels around randomly"),
    NULL);
}

#endif
