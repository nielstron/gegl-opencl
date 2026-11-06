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
 * Copyright 2010 Barak Itkin <lightningismyname@gmail.org>
 *
 * Based on "Whirl and Pinch" GIMP plugin
 * Copyright (C) 1997 Federico Mena Quintero
 * Copyright (C) 1997 Scott Goehring
 *
 * The workshop/mirrors.c operation by Alexia Death and Øyvind Kolås
 * was used as a template for this op file.
 */

/* TODO: Find some better algorithm to calculate the roi for each dest
 *       rectangle. Right now it simply asks for the entire image...
 */

#include "config.h"
#include <glib/gi18n-lib.h>


#ifdef GEGL_PROPERTIES

property_double (whirl, _("Whirl"), 90.0)
    description (_("Whirl angle (degrees)"))
    ui_range    (-720, 720)
    ui_meta     ("unit", "degree")
    ui_meta     ("direction", "ccw")

property_double (pinch, _("Pinch"), 0.0)
    description (_("Pinch amount"))
    value_range (-1.0, 1.0)

property_double (radius, _("Radius"), 1.0)
    description(_("Radius (1.0 is the largest circle that fits in the "
               "image, and 2.0 goes all the way to the corners)"))
    value_range (0.0, 2.0)

#else

#define GEGL_OP_FILTER
#define GEGL_OP_C_SOURCE whirl-pinch.c
#define GEGL_OP_NAME     whirl_pinch

#include "gegl-op.h"

/* This function is a slightly modified version from the one in the
 * original plugin
 */
static gboolean
calc_undistorted_coords (gdouble  wx,
                         gdouble  wy,
                         gdouble  cen_x,
                         gdouble  cen_y,
                         gdouble  scale_x,
                         gdouble  scale_y,
                         gdouble  whirl,
                         gdouble  pinch,
                         gdouble  wpradius,
                         gdouble *x,
                         gdouble *y)
{
  gdouble  dx, dy;
  gdouble  d, factor;
  gdouble  dist;
  gdouble  ang, sina, cosa;
  gboolean inside;

  gdouble  radius = MAX(cen_x, cen_y);
  gdouble  radius2 = radius * radius * wpradius;
  /* Distances to center, scaled */

  dx = (wx - cen_x) * scale_x;
  dy = (wy - cen_y) * scale_y;

  /* Distance^2 to center of *circle* (scaled ellipse) */

  d = dx * dx + dy * dy;

  /*  If we are inside circle, then distort.
   *  Else, just return the same position
   */

  inside = (d < radius2);

  /* If d is 0, then we are exactly at the center, so the transform has
   * no effect. The original version of the gimp plugin simply created
   * an offset of the original points to avoid this issue, but that is
   * less accurate...
   */
  if (inside && d > 0)
    {
      dist = sqrt(d / wpradius) / radius;

      /* Pinch */

      factor = pow (sin (G_PI_2 * dist), -pinch);

      dx *= factor;
      dy *= factor;

      /* Whirl */

      factor = 1.0 - dist;

      ang = whirl * factor * factor;

      sina = sin (ang);
      cosa = cos (ang);

      *x = (cosa * dx - sina * dy) / scale_x + cen_x;
      *y = (sina * dx + cosa * dy) / scale_y + cen_y;
    }
  else
    {
      *x = wx;
      *y = wy;
    }

  return inside;
}

/* Apply the actual transform */

static void
apply_whirl_pinch (gdouble              whirl,
                   gdouble              pinch,
                   gdouble              radius,
                   gdouble              cen_x,
                   gdouble              cen_y,
                   const Babl          *format,
                   GeglBuffer          *src,
                   GeglRectangle       *in_boundary,
                   GeglBuffer          *dst,
                   GeglRectangle       *boundary,
                   const GeglRectangle *roi,
                   gint                 level)
{
  gfloat *dst_buf;
  gint row, col;
  gdouble scale_x, scale_y;
  gdouble cx, cy;
  GeglSampler *sampler;

  /* Get buffer in which to place dst pixels. */
  dst_buf = g_new0 (gfloat, roi->width * roi->height * 4);

  whirl = whirl * G_PI / 180;

  scale_x = 1.0;
  scale_y = (gdouble) in_boundary->width / in_boundary->height;
  sampler = gegl_buffer_sampler_new_at_level (src, format,
                                     GEGL_SAMPLER_NOHALO, level);

  for (row = 0; row < roi->height; row++) {
    for (col = 0; col < roi->width; col++) {
        GeglBufferMatrix2 scale;
#define gegl_unmap(u,v,du,dv) \
        { \
          calc_undistorted_coords (u, v,\
                                   cen_x, cen_y,\
                                   scale_x, scale_y,\
                                   whirl, pinch, radius,\
                                   &cx, &cy);\
          du=cx;dv=cy;\
        }
        gegl_sampler_compute_scale (scale, roi->x + col, roi->y + row);
        gegl_unmap (roi->x + col, roi->y + row, cx, cy);

        gegl_sampler_get (sampler, cx, cy, &scale, &dst_buf[(row * roi->width + col) * 4], GEGL_ABYSS_CLAMP);
    } /* for */
  } /* for */

  /* Store dst pixels. */
  gegl_buffer_set (dst, roi, 0, format, dst_buf, GEGL_AUTO_ROWSTRIDE);

  g_free (dst_buf);
  g_object_unref (sampler);
}

/*****************************************************************************/

/* Compute the input rectangle required to compute the specified region of interest (roi).
 */
static GeglRectangle
get_required_for_output (GeglOperation       *operation,
                         const gchar         *input_pad,
                         const GeglRectangle *roi)
{
  const GeglRectangle *in_rect =
            gegl_operation_source_get_bounding_box (operation, "input");

  if (! in_rect || gegl_rectangle_is_infinite_plane (in_rect))
    {
      return *roi;
    }

  return *in_rect;
}

/* Specify the input and output buffer formats.
 */
static void
prepare (GeglOperation *operation)
{
  const Babl *space = gegl_operation_get_source_space (operation, "input");
  gegl_operation_set_format (operation, "input", babl_format_with_space ("RaGaBaA float", space));
  gegl_operation_set_format (operation, "output", babl_format_with_space ("RaGaBaA float", space));
}

#include "opencl/gegl-cl.h"
#include "gegl-buffer-cl-iterator.h"

#include "opencl/whirl-pinch.cl.h"

static GeglClRunData *cl_data = NULL;

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

/* Implements apply_whirl_pinch () with the NoHalo sampler on the GPU.  The
 * sampler uses GEGL_ABYSS_CLAMP, so every sample lands inside the input's
 * abyss rectangle, which is uploaded as a whole.
 */
static gboolean
cl_process (GeglOperation       *operation,
            GeglBuffer          *input,
            GeglBuffer          *output,
            const GeglRectangle *roi,
            const GeglRectangle *boundary,
            gdouble              whirl,
            gdouble              pinch,
            gdouble              radius)
{
  const Babl           *format = gegl_operation_get_format (operation, "output");
  const GeglRectangle  *abyss  = gegl_buffer_get_abyss (input);
  GeglBufferClIterator *i;
  cl_mem                in_tex = NULL;
  cl_int                cl_err = 0;
  gboolean              err;
  cl_int4               srect;
  cl_int4               abyss4;
  cl_int                policy  = GEGL_ABYSS_CLAMP;
  cl_float              cen_x   = boundary->width  / 2.0;
  cl_float              cen_y   = boundary->height / 2.0;
  cl_float              scale_x = 1.0;
  cl_float              scale_y = (gdouble) boundary->width / boundary->height;
  cl_float              f_whirl = whirl * G_PI / 180;
  cl_float              f_pinch = pinch;
  cl_float              f_radius = radius;

  if (abyss->width < 1 || abyss->height < 1 ||
      (gdouble) abyss->width * abyss->height > (1 << 26))
    return FALSE;

  if (!cl_data)
    {
      const char *kernel_name[] = {"whirl_pinch", NULL};
      cl_data = gegl_cl_compile_and_build (whirl_pinch_cl_source, kernel_name);
    }

  if (!cl_data)
    return FALSE;

  srect.s[0] = abyss4.s[0] = abyss->x;
  srect.s[1] = abyss4.s[1] = abyss->y;
  srect.s[2] = abyss4.s[2] = abyss->width;
  srect.s[3] = abyss4.s[3] = abyss->height;

  in_tex = cl_upload_rect (input, abyss, format);
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

      cl_err = gegl_cl_set_kernel_args (cl_data->kernel[0],
                                        sizeof (cl_mem),   &in_tex,
                                        sizeof (cl_int4),  &srect,
                                        sizeof (cl_int4),  &abyss4,
                                        sizeof (cl_int),   &policy,
                                        sizeof (cl_mem),   &i->tex[0],
                                        sizeof (cl_int),   &i->roi[0].x,
                                        sizeof (cl_int),   &i->roi[0].y,
                                        sizeof (cl_float), &cen_x,
                                        sizeof (cl_float), &cen_y,
                                        sizeof (cl_float), &scale_x,
                                        sizeof (cl_float), &scale_y,
                                        sizeof (cl_float), &f_whirl,
                                        sizeof (cl_float), &f_pinch,
                                        sizeof (cl_float), &f_radius,
                                        NULL);
      CL_CHECK;

      cl_err = gegl_clEnqueueNDRangeKernel (gegl_cl_get_command_queue (),
                                            cl_data->kernel[0], 2,
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

/* Perform the specified operation.
 */
static gboolean
process (GeglOperation       *operation,
         GeglBuffer          *input,
         GeglBuffer          *output,
         const GeglRectangle *result,
         gint                 level)
{
  GeglProperties *o        = GEGL_PROPERTIES (operation);
  GeglRectangle   boundary = gegl_operation_get_bounding_box (operation);
  const Babl     *format   = gegl_operation_get_format (operation, "output");

  if (level == 0 && gegl_operation_use_opencl (operation) &&
      cl_process (operation, input, output, result, &boundary,
                  o->whirl, o->pinch, o->radius))
    return TRUE;

  apply_whirl_pinch (o->whirl,
                     o->pinch,
                     o->radius,
                     boundary.width / 2.0,
                     boundary.height / 2.0,
                     format,
                     input,
                     &boundary,
                     output,
                     &boundary,
                     result,
                     level);
  return TRUE;
}

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

  if (in_rect && gegl_rectangle_is_infinite_plane (in_rect))
    {
      gpointer in = gegl_operation_context_get_object (context, "input");
      gegl_operation_context_take_object (context, "output",
                                          g_object_ref (G_OBJECT (in)));
      return TRUE;
    }

  operation_class = GEGL_OPERATION_CLASS (gegl_op_parent_class);

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
  operation_class->get_required_for_output = get_required_for_output;
  operation_class->opencl_support          = TRUE;

  gegl_operation_class_set_keys (operation_class,
    "name",               "gegl:whirl-pinch",
    "title",              _("Whirl Pinch"),
    "categories",         "distort:map",
    "license",            "GPL3+",
    "position-dependent", "true",
    "reference-hash",     "cd55e78cba530fc94254cf59425e941e",
    "description", _("Distort an image by whirling and pinching"),
    NULL);
}

#endif
