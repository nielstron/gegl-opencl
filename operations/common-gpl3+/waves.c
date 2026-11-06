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
 * Copyright 2011 Audrey Mandet <mandet.audrey@gmail.com>
 * Copyright (c) 1997 Eric L. Hernes (erich@rrnet.com), Stephen Norris
 */

#include "config.h"
#include <glib/gi18n-lib.h>

#ifdef GEGL_PROPERTIES

property_double (x, _("Center X"), 0.5)
    ui_range   (0.0, 1.0)
    ui_meta    ("unit", "relative-coordinate")
    ui_meta    ("axis", "x")

property_double (y, _("Center Y"), 0.5)
    ui_range   (0.0, 1.0)
    ui_meta    ("unit", "relative-coordinate")
    ui_meta    ("axis", "y")

property_double (amplitude, _("Amplitude"), 25.0)
    description(_("Amplitude of the ripple"))
    value_range (0.0, 1000.0)

property_double (period, _("Period"), 100.0)
    description(_("Period (wavelength) of the ripple"))
    value_range (0.1, 1000.0)

property_double (phi, _("Phase shift"), 0)
    value_range (-1, 1)

property_double (aspect, _("Aspect ratio"), 1.0)
    value_range (0.1, 10.0)  /* XXX: needs better ui type */

property_enum (sampler_type, _("Resampling method"),
    GeglSamplerType, gegl_sampler_type, GEGL_SAMPLER_CUBIC)
    description(_("Mathematical method for reconstructing pixel values"))

property_boolean (clamp, _("Clamp deformation"), FALSE)
    description(_("Limit deformation in the image area."))

#else

#define GEGL_OP_AREA_FILTER
#define GEGL_OP_C_SOURCE waves.c
#define GEGL_OP_NAME     waves

#include "gegl-op.h"
#include <stdio.h>

static void
prepare (GeglOperation *operation)
{
  GeglOperationAreaFilter *op_area = GEGL_OPERATION_AREA_FILTER (operation);
  GeglProperties          *o       = GEGL_PROPERTIES (operation);
  const Babl              *space   = gegl_operation_get_source_space (operation, "input");

  op_area->left   = o->amplitude;
  op_area->right  = o->amplitude;
  op_area->top    = o->amplitude;
  op_area->bottom = o->amplitude;

  gegl_operation_set_format (operation, "input",  babl_format_with_space ("RGBA float", space));
  gegl_operation_set_format (operation, "output", babl_format_with_space ("RGBA float", space));
}

static GeglAbyssPolicy
get_abyss_policy (GeglOperation *operation,
                  const gchar   *input_pad)
{
  GeglProperties *o = GEGL_PROPERTIES (operation);

  return o->clamp ? GEGL_ABYSS_CLAMP : GEGL_ABYSS_NONE;
}

#include "opencl/gegl-cl.h"
#include "gegl-buffer-cl-iterator.h"

#include "opencl/waves.cl.h"

static GeglClRunData *cl_data  = NULL;
static cl_mem         cl_table = NULL;

/* must match the kernel */
#define WAVES_TABLE_SIZE 256

/* @value as an unevaluated sum of two floats */
static cl_float2
cl_split_double (gdouble value)
{
  cl_float2 v;

  v.s[0] = value;
  v.s[1] = value - (gdouble) v.s[0];

  return v;
}

/* sin and cos of 2 pi k / WAVES_TABLE_SIZE, as (sin hi, sin lo, cos hi,
 * cos lo) */
static cl_mem
cl_create_table (void)
{
  cl_float4 table[WAVES_TABLE_SIZE];
  cl_int    cl_err = 0;
  cl_mem    tex;
  gint      k;

  for (k = 0; k < WAVES_TABLE_SIZE; k++)
    {
      cl_float2 s = cl_split_double (sin (2.0 * G_PI * k / WAVES_TABLE_SIZE));
      cl_float2 c = cl_split_double (cos (2.0 * G_PI * k / WAVES_TABLE_SIZE));

      table[k].s[0] = s.s[0];
      table[k].s[1] = s.s[1];
      table[k].s[2] = c.s[0];
      table[k].s[3] = c.s[1];
    }

  tex = gegl_clCreateBuffer (gegl_cl_get_context (),
                             CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY,
                             sizeof (table), table, &cl_err);
  if (cl_err != CL_SUCCESS)
    return NULL;

  return tex;
}

/* Copies @rect of @buffer, converted to @format, into a new OpenCL buffer. */
static cl_mem
cl_upload_rect (GeglBuffer          *buffer,
                const GeglRectangle *rect,
                const Babl          *format)
{
  cl_int   cl_err = 0;
  cl_mem   tex    = NULL;
  size_t   size   = (size_t) rect->width * rect->height *
                    babl_format_get_bytes_per_pixel (format);
  gpointer data;

  tex = gegl_clCreateBuffer (gegl_cl_get_context (),
                             CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY,
                             size, NULL, &cl_err);
  CL_CHECK;

  data = gegl_clEnqueueMapBuffer (gegl_cl_get_command_queue (), tex, CL_TRUE,
                                  CL_MAP_WRITE, 0, size,
                                  0, NULL, NULL, &cl_err);
  CL_CHECK;

  gegl_buffer_get (buffer, rect, 1.0, format, data,
                   GEGL_AUTO_ROWSTRIDE, GEGL_ABYSS_NONE);

  cl_err = gegl_clEnqueueUnmapMemObject (gegl_cl_get_command_queue (), tex,
                                         data, 0, NULL, NULL);
  CL_CHECK;

  return tex;

error:
  if (tex)
    gegl_clReleaseMemObject (tex);

  return NULL;
}

/* Computes the part of the input that has to be uploaded so that the
 * kernel's fetch () can serve every pixel in @needed the way GeglSampler
 * does, given the input's @abyss rectangle and abyss @policy.
 */
static gboolean
cl_get_source_rect (const GeglRectangle *needed,
                    const GeglRectangle *abyss,
                    GeglAbyssPolicy      policy,
                    GeglRectangle       *src_rect)
{
  if (gegl_rectangle_is_infinite_plane (abyss))
    {
      if (policy == GEGL_ABYSS_LOOP)
        return FALSE;

      *src_rect = *needed;
    }
  else if (abyss->width < 1 || abyss->height < 1)
    {
      return FALSE;
    }
  else if (policy == GEGL_ABYSS_LOOP)
    {
      if (gegl_rectangle_contains (abyss, needed))
        *src_rect = *needed;
      else
        *src_rect = *abyss;
    }
  else if (policy == GEGL_ABYSS_CLAMP)
    {
      /* where @needed lands after clamping to @abyss */
      gint x0 = CLAMP (needed->x, abyss->x, abyss->x + abyss->width - 1);
      gint y0 = CLAMP (needed->y, abyss->y, abyss->y + abyss->height - 1);
      gint x1 = CLAMP (needed->x + needed->width - 1,
                       abyss->x, abyss->x + abyss->width - 1);
      gint y1 = CLAMP (needed->y + needed->height - 1,
                       abyss->y, abyss->y + abyss->height - 1);

      *src_rect = *GEGL_RECTANGLE (x0, y0, x1 - x0 + 1, y1 - y0 + 1);
    }
  else if (! gegl_rectangle_intersect (src_rect, needed, abyss))
    {
      /* everything is abyss, the uploaded pixel is never read */
      *src_rect = *GEGL_RECTANGLE (abyss->x, abyss->y, 1, 1);
    }

  return (gdouble) src_rect->width * src_rect->height <= (1 << 26);
}

static gboolean
cl_process (GeglOperation       *operation,
            GeglBuffer          *input,
            GeglBuffer          *output,
            const GeglRectangle *roi,
            GeglAbyssPolicy      abyss_policy,
            gdouble              px_x,
            gdouble              px_y,
            gdouble              scalex,
            gdouble              scaley)
{
  GeglProperties       *o      = GEGL_PROPERTIES (operation);
  const Babl           *format = gegl_operation_get_format (operation, "output");
  const Babl           *space  = babl_format_get_space (format);
  const Babl           *src_format;
  const GeglRectangle  *abyss  = gegl_buffer_get_abyss (input);
  GeglRectangle         needed;
  GeglRectangle         src_rect;
  GeglBufferClIterator *i;
  cl_kernel             kernel;
  cl_mem                in_tex = NULL;
  cl_int                cl_err = 0;
  gboolean              err;
  gint                  margin;
  cl_int4               srect;
  cl_int4               abyss4;
  cl_int                policy      = abyss_policy;
  cl_int                unpremultiply;
  /* the double precision parameters, as hi + lo */
  cl_float2             center_x    = cl_split_double (px_x);
  cl_float2             center_y    = cl_split_double (px_y);
  cl_float2             scale_x     = cl_split_double (scalex);
  cl_float2             scale_y     = cl_split_double (scaley);
  cl_float2             amplitude   = cl_split_double (o->amplitude);
  cl_float2             period      = cl_split_double (o->period);
  cl_float2             phi         = cl_split_double (o->phi);

  if (format != babl_format_with_space ("RGBA float", space))
    return FALSE;

  switch (o->sampler_type)
    {
    case GEGL_SAMPLER_NEAREST:
      /* the nearest sampler converts pixels straight to the op's format */
      src_format    = format;
      unpremultiply = 0;
      break;
    case GEGL_SAMPLER_LINEAR:
    case GEGL_SAMPLER_CUBIC:
    case GEGL_SAMPLER_NOHALO:
      src_format    = babl_format_with_space ("RaGaBaA float", space);
      unpremultiply = 1;
      break;
    default:
      return FALSE;
    }

  /* samples are at most amplitude + 1 pixels away (scale >= 1), and the
   * samplers read up to two pixels beyond the sampled one */
  margin = ceil (fabs (o->amplitude)) + 4;
  needed = *roi;
  needed.x      -= margin;
  needed.y      -= margin;
  needed.width  += 2 * margin;
  needed.height += 2 * margin;

  if (! cl_get_source_rect (&needed, abyss, policy, &src_rect))
    return FALSE;

  if (!cl_data)
    {
      const char *kernel_name[] = {"waves_nearest", "waves_linear",
                                   "waves_cubic", "waves_nohalo", NULL};
      cl_data = gegl_cl_compile_and_build (waves_cl_source, kernel_name);
    }

  if (!cl_data)
    return FALSE;

  if (!cl_table)
    cl_table = cl_create_table ();

  if (!cl_table)
    return FALSE;

  switch (o->sampler_type)
    {
    case GEGL_SAMPLER_NEAREST: kernel = cl_data->kernel[0]; break;
    case GEGL_SAMPLER_LINEAR:  kernel = cl_data->kernel[1]; break;
    case GEGL_SAMPLER_CUBIC:   kernel = cl_data->kernel[2]; break;
    default:                   kernel = cl_data->kernel[3]; break;
    }

  srect.s[0]  = src_rect.x;
  srect.s[1]  = src_rect.y;
  srect.s[2]  = src_rect.width;
  srect.s[3]  = src_rect.height;
  abyss4.s[0] = abyss->x;
  abyss4.s[1] = abyss->y;
  abyss4.s[2] = abyss->width;
  abyss4.s[3] = abyss->height;

  in_tex = cl_upload_rect (input, &src_rect, src_format);
  if (!in_tex)
    return FALSE;

  i = gegl_buffer_cl_iterator_new (output, roi, format, GEGL_CL_BUFFER_WRITE);

  while (gegl_buffer_cl_iterator_next (i, &err))
    {
      size_t global_ws[2];

      if (err)
        goto error_iter;

      global_ws[0] = i->roi[0].width;
      global_ws[1] = i->roi[0].height;

      cl_err = gegl_cl_set_kernel_args (kernel,
                                        sizeof (cl_mem),   &in_tex,
                                        sizeof (cl_int4),  &srect,
                                        sizeof (cl_int4),  &abyss4,
                                        sizeof (cl_int),   &policy,
                                        sizeof (cl_mem),   &i->tex[0],
                                        sizeof (cl_int),   &i->roi[0].x,
                                        sizeof (cl_int),   &i->roi[0].y,
                                        sizeof (cl_int),    &unpremultiply,
                                        sizeof (cl_mem),    &cl_table,
                                        sizeof (cl_float2), &center_x,
                                        sizeof (cl_float2), &center_y,
                                        sizeof (cl_float2), &scale_x,
                                        sizeof (cl_float2), &scale_y,
                                        sizeof (cl_float2), &amplitude,
                                        sizeof (cl_float2), &period,
                                        sizeof (cl_float2), &phi,
                                        NULL);
      CL_CHECK;

      cl_err = gegl_clEnqueueNDRangeKernel (gegl_cl_get_command_queue (),
                                            kernel, 2,
                                            NULL, global_ws, NULL,
                                            0, NULL, NULL);
      CL_CHECK;
    }

  gegl_clReleaseMemObject (in_tex);

  return TRUE;

error:
  gegl_buffer_cl_iterator_stop (i);
error_iter:
  gegl_clReleaseMemObject (in_tex);

  return FALSE;
}

static gboolean
process (GeglOperation       *operation,
         GeglBuffer          *input,
         GeglBuffer          *output,
         const GeglRectangle *result,
         gint                 level)
{
  GeglProperties     *o       = GEGL_PROPERTIES (operation);
  const Babl         *format  = gegl_operation_get_format (operation, "output");
  GeglSampler        *sampler = gegl_buffer_sampler_new_at_level (input, format,
                                                         o->sampler_type,
                                                         level);
  GeglRectangle      *in_extent = gegl_operation_source_get_bounding_box (operation, "input");
  GeglBufferIterator *iter;

  GeglAbyssPolicy abyss = get_abyss_policy (operation, "input");

  gdouble px_x = gegl_coordinate_relative_to_pixel (o->x, in_extent->width);
  gdouble px_y = gegl_coordinate_relative_to_pixel (o->y, in_extent->height);

  gdouble scalex;
  gdouble scaley;

  if (o->aspect > 1.0)
    {
      scalex = 1.0;
      scaley = o->aspect;
    }
  else if (o->aspect < 1.0)
    {
      scalex = 1.0 / o->aspect;
      scaley = 1.0;
    }
  else
    {
      scalex = 1.0;
      scaley = 1.0;
    }

  if (level == 0 && gegl_operation_use_opencl (operation) &&
      cl_process (operation, input, output, result, abyss,
                  px_x, px_y, scalex, scaley))
    {
      g_object_unref (sampler);
      return TRUE;
    }

  iter = gegl_buffer_iterator_new (output, result, 0, format,
                                   GEGL_ACCESS_WRITE, GEGL_ABYSS_NONE, 1);

  while (gegl_buffer_iterator_next (iter))
    {
      gint x = result->x;
      gint y = result->y;
      gfloat *out_pixel = iter->items[0].data;

      for (y = iter->items[0].roi.y; y < iter->items[0].roi.y + iter->items[0].roi.height; ++y)
        for (x = iter->items[0].roi.x; x < iter->items[0].roi.x + iter->items[0].roi.width; ++x)
          {
            gdouble radius;
            gdouble shift;
            gdouble dx;
            gdouble dy;
            gdouble ux;
            gdouble uy;

            dx = (x - px_x) * scalex;
            dy = (y - px_y) * scaley;

            if (!dx && !dy)
              radius = 0.000001;
            else
              radius = sqrt (dx * dx + dy * dy);

            shift = o->amplitude * sin (2.0 * G_PI * radius / o->period +
                                        2.0 * G_PI * o->phi);

            ux = dx / radius;
            uy = dy / radius;

            gegl_sampler_get (sampler,
                              x + (shift + ux) / scalex,
                              y + (shift + uy) / scaley,
                              NULL,
                              out_pixel,
                              abyss);

            out_pixel += 4;
          }
    }

  g_object_unref (sampler);

  return TRUE;
}

static void
gegl_op_class_init (GeglOpClass *klass)
{
  GeglOperationClass           *operation_class;
  GeglOperationFilterClass     *filter_class;
  GeglOperationAreaFilterClass *area_class;

  operation_class = GEGL_OPERATION_CLASS (klass);
  filter_class    = GEGL_OPERATION_FILTER_CLASS (klass);
  area_class      = GEGL_OPERATION_AREA_FILTER_CLASS (klass);

  operation_class->prepare     = prepare;
  filter_class->process        = process;
  area_class->get_abyss_policy = get_abyss_policy;
  operation_class->opencl_support = TRUE;

  gegl_operation_class_set_keys (operation_class,
    "name",               "gegl:waves",
    "title",              _("Waves"),
    "categories",         "distort",
    "position-dependent", "true",
    "license",            "GPL3+",
    "reference-hash",     "c7e6bdb018b7597c31770fbb935d8ecd",
    "description", _("Distort the image with waves"),
    NULL);
}

#endif
