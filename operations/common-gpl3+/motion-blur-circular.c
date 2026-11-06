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
 * Copyright (C) 2013 Téo Mazars  <teo.mazars@ensimag.fr>
 *
 * This operation is inspired by and uses parts of blur-motion.c
 * from GIMP 2.8.4:
 *
 * Copyright (C) 1995 Spencer Kimball and Peter Mattis
 * Copyright (C) 1996 Torsten Martinsen       <torsten@danbbs.dk>
 * Copyright (C) 1996 Federico Mena Quintero  <federico@nuclecu.unam.mx>
 * Copyright (C) 1996 Heinz W. Werntges       <quartic@polloux.fciencias.unam.mx>
 * Copyright (C) 1997 Daniel Skarda           <0rfelyus@atrey.karlin.mff.cuni.cz>
 * Copyright (C) 2007 Joerg Gittinger         <sw@gittingerbox.de>
 *
 * This operation is also inspired by GEGL's blur-motion-linear.c :
 *
 * Copyright (C) 2006 Øyvind Kolås  <pippin@gimp.org>
 */


#include "config.h"

#include <glib/gi18n-lib.h>

#ifdef GEGL_PROPERTIES

property_double (center_x, _("Center X"), 0.5)
    ui_range    (0.0, 1.0)
    ui_meta     ("unit", "relative-coordinate")
    ui_meta     ("axis", "x")

property_double (center_y, _("Center Y"), 0.5)
    ui_range    (0.0, 1.0)
    ui_meta     ("unit", "relative-coordinate")
    ui_meta     ("axis", "y")

/* FIXME: With a large angle, we lose AreaFilter's flavours */
property_double (angle, _("Angle"), 5.0)
    description (_("Rotation blur angle. A large angle may take some time to render"))
    value_range (0.0, 360.0)
    ui_meta     ("unit", "degree")

#else

#define GEGL_OP_AREA_FILTER
#define GEGL_OP_NAME     motion_blur_circular
#define GEGL_OP_C_SOURCE motion-blur-circular.c

#include "gegl-op.h"

#define SQR(c) ((c) * (c))

#define NOMINAL_NUM_IT  100
#define SQRT_2          1.41

static void
prepare (GeglOperation *operation)
{
  GeglOperationAreaFilter *op_area = GEGL_OPERATION_AREA_FILTER (operation);
  GeglProperties          *o       = GEGL_PROPERTIES (operation);
  gdouble                  angle   = o->angle * G_PI / 180.0;
  const Babl              *space   = gegl_operation_get_source_space (operation, "input");
  GeglRectangle           *whole_region;

  whole_region = gegl_operation_source_get_bounding_box (operation, "input");

  if (whole_region && ! gegl_rectangle_is_infinite_plane (whole_region))
    {
      gdouble center_x = gegl_coordinate_relative_to_pixel (o->center_x,
                                                            whole_region->width);
      gdouble center_y = gegl_coordinate_relative_to_pixel (o->center_y,
                                                            whole_region->height);
      gdouble maxr_x = MAX (fabs (center_x - whole_region->x),
                            fabs (center_x - whole_region->x - whole_region->width));
      gdouble maxr_y = MAX (fabs (center_y - whole_region->y),
                            fabs (center_y - whole_region->y - whole_region->height));

      if (angle >= G_PI)
        angle = G_PI;

      op_area->left = op_area->right
        = ceil (maxr_y * sin (angle / 2.0)) + 1;

      op_area->top = op_area->bottom
        = ceil (maxr_x * sin (angle / 2.0)) + 1;
    }
  else
    {
      op_area->left   =
      op_area->right  =
      op_area->top    =
      op_area->bottom = 0;
    }

  gegl_operation_set_format (operation, "input",  babl_format_with_space ("RaGaBaA float", space));
  gegl_operation_set_format (operation, "output", babl_format_with_space ("RaGaBaA float", space));
}


/* Computes how far the samples of a pixel inside the input extent can
 * be from it.  The samples of a pixel are its offset from the center
 * rotated by theta, |theta| <= angle / 2, which moves it by
 * (xr * (cos (theta) - 1) - yr * sin (theta),
 *  xr * sin (theta) + yr * (cos (theta) - 1)).
 * One pixel is added for the bilinear interpolation.
 */
static gboolean
get_sample_distance (GeglOperation *operation,
                     gint          *dx,
                     gint          *dy)
{
  GeglProperties *o = GEGL_PROPERTIES (operation);
  GeglRectangle  *whole_region;
  gdouble         angle = o->angle * G_PI / 180.0;
  gdouble         center_x, center_y;
  gdouble         maxr_x, maxr_y;
  gdouble         half, max_sin, max_cos;

  whole_region = gegl_operation_source_get_bounding_box (operation, "input");

  if (! whole_region || gegl_rectangle_is_infinite_plane (whole_region))
    {
      *dx = *dy = 0;
      return FALSE;
    }

  center_x = gegl_coordinate_relative_to_pixel (o->center_x,
                                                whole_region->width);
  center_y = gegl_coordinate_relative_to_pixel (o->center_y,
                                                whole_region->height);
  maxr_x = MAX (fabs (center_x - whole_region->x),
                fabs (center_x - whole_region->x - whole_region->width));
  maxr_y = MAX (fabs (center_y - whole_region->y),
                fabs (center_y - whole_region->y - whole_region->height));

  half    = MIN (angle / 2.0, G_PI);
  max_sin = half >= G_PI_2 ? 1.0 : sin (half);
  max_cos = 1.0 - cos (half);

  *dx = MIN (ceil (maxr_x * max_cos + maxr_y * max_sin),
             whole_region->width) + 1;
  *dy = MIN (ceil (maxr_y * max_cos + maxr_x * max_sin),
             whole_region->height) + 1;

  return TRUE;
}

/* Computes the area around @rect that its samples can come from, so that
 * no sample is clamped to the edge of the input rectangle, which would
 * make the result depend on how the output is split in chunks.  Samples
 * outside the input extent are skipped, so the area is clipped to it (plus
 * one pixel for the bilinear interpolation).  The clipping uses the far
 * edge of @rect, so that the area is also large enough for any part of
 * @rect.
 */
static void
get_sample_area (GeglOperation       *operation,
                 const GeglRectangle *rect,
                 gint                *left,
                 gint                *right,
                 gint                *top,
                 gint                *bottom)
{
  GeglRectangle *whole_region;
  gint           dx, dy;

  if (! get_sample_distance (operation, &dx, &dy))
    {
      *left = *right = *top = *bottom = 0;
      return;
    }

  whole_region = gegl_operation_source_get_bounding_box (operation, "input");

  *left   = CLAMP (rect->x + rect->width - 1 - whole_region->x, 0, dx);
  *right  = CLAMP (whole_region->x + whole_region->width - rect->x, 0, dx);
  *top    = CLAMP (rect->y + rect->height - 1 - whole_region->y, 0, dy);
  *bottom = CLAMP (whole_region->y + whole_region->height - rect->y, 0, dy);
}

static GeglRectangle
get_required_for_output (GeglOperation       *operation,
                         const gchar         *input_pad,
                         const GeglRectangle *roi)
{
  GeglRectangle rect = *roi;
  gint          left, right, top, bottom;

  get_sample_area (operation, roi, &left, &right, &top, &bottom);

  rect.x      -= left;
  rect.y      -= top;
  rect.width  += left + right;
  rect.height += top + bottom;

  return rect;
}

static GeglRectangle
get_invalidated_by_change (GeglOperation       *operation,
                           const gchar         *input_pad,
                           const GeglRectangle *input_region)
{
  GeglRectangle rect = *input_region;
  gint          dx, dy;

  if (! get_sample_distance (operation, &dx, &dy))
    return gegl_operation_get_bounding_box (operation);

  rect.x      -= dx;
  rect.y      -= dy;
  rect.width  += 2 * dx;
  rect.height += 2 * dy;

  return rect;
}

static inline gfloat *
get_pixel_color (gfloat              *in_buf,
                 const GeglRectangle *rect,
                 gint                 x,
                 gint                 y)
{
  gint ix = x - rect->x;
  gint iy = y - rect->y;

  ix = CLAMP (ix, 0, rect->width  - 1);
  iy = CLAMP (iy, 0, rect->height - 1);

  return &in_buf[(iy * rect->width + ix) * 4];
}

#include "opencl/gegl-cl.h"
#include "gegl-buffer-cl-iterator.h"

#include "opencl/motion-blur-circular.cl.h"

static GeglClRunData *cl_data = NULL;

static gboolean
cl_motion_blur_circular (cl_mem               in,
                         cl_mem               out,
                         const GeglRectangle *src_rect,
                         const GeglRectangle *dst_rect,
                         const GeglRectangle *whole_region,
                         gdouble              angle,
                         gdouble              center_x,
                         gdouble              center_y)
{
  cl_int    cl_err = 0;
  size_t    global_ws[2];
  cl_float  a;
  cl_float2 arc_factor_sq, cx, cy;
  gdouble   arc_factor = angle * SQRT_2;

  if (!cl_data)
    {
      const char *kernel_name[] = { "cl_motion_blur_circular", NULL };
      cl_data = gegl_cl_compile_and_build (motion_blur_circular_cl_source,
                                           kernel_name);
    }

  if (!cl_data)
    return TRUE;

  /* pass doubles that need full precision as hi + lo pairs */
  a = (cl_float) angle;
  arc_factor_sq.s[0] = (cl_float) (arc_factor * arc_factor);
  arc_factor_sq.s[1] = (cl_float) (arc_factor * arc_factor - arc_factor_sq.s[0]);
  cx.s[0] = (cl_float) center_x;
  cx.s[1] = (cl_float) (center_x - cx.s[0]);
  cy.s[0] = (cl_float) center_y;
  cy.s[1] = (cl_float) (center_y - cy.s[0]);
  cl_err = gegl_cl_set_kernel_args (cl_data->kernel[0],
                                    sizeof(cl_mem),   &in,
                                    sizeof(cl_mem),   &out,
                                    sizeof(cl_int),   &src_rect->width,
                                    sizeof(cl_int),   &src_rect->height,
                                    sizeof(cl_int),   &src_rect->x,
                                    sizeof(cl_int),   &src_rect->y,
                                    sizeof(cl_int),   &dst_rect->x,
                                    sizeof(cl_int),   &dst_rect->y,
                                    sizeof(cl_int),   &whole_region->width,
                                    sizeof(cl_int),   &whole_region->height,
                                    sizeof(cl_int),   &whole_region->x,
                                    sizeof(cl_int),   &whole_region->y,
                                    sizeof(cl_float),  &a,
                                    sizeof(cl_float2), &arc_factor_sq,
                                    sizeof(cl_float2), &cx,
                                    sizeof(cl_float2), &cy,
                                    NULL);
  CL_CHECK;

  global_ws[0] = dst_rect->width;
  global_ws[1] = dst_rect->height;
  cl_err = gegl_clEnqueueNDRangeKernel (gegl_cl_get_command_queue (),
                                        cl_data->kernel[0], 2,
                                        NULL, global_ws, NULL,
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
            const GeglRectangle *whole_region,
            gdouble              angle,
            gdouble              center_x,
            gdouble              center_y)
{
  GeglBufferClIterator *i;
  const Babl *in_format  = gegl_operation_get_format (operation, "input");
  const Babl *out_format = gegl_operation_get_format (operation, "output");
  gint        err;
  gint        read;
  gint        left, right, top, bottom;

  get_sample_area (operation, result, &left, &right, &top, &bottom);

  i = gegl_buffer_cl_iterator_new (output,
                                   result,
                                   out_format,
                                   GEGL_CL_BUFFER_WRITE);

  read = gegl_buffer_cl_iterator_add_2 (i,
                                        input,
                                        result,
                                        in_format,
                                        GEGL_CL_BUFFER_READ,
                                        left, right, top, bottom,
                                        GEGL_ABYSS_NONE);

  while (gegl_buffer_cl_iterator_next (i, &err))
    {
      if (err)
        return FALSE;

      err = cl_motion_blur_circular (i->tex[read],
                                     i->tex[0],
                                     &i->roi[read],
                                     &i->roi[0],
                                     whole_region,
                                     angle,
                                     center_x,
                                     center_y);

      if (err)
        return FALSE;
    }

  return TRUE;
}

static gboolean
process (GeglOperation       *operation,
         GeglBuffer          *input,
         GeglBuffer          *output,
         const GeglRectangle *roi,
         gint                 level)
{
  GeglProperties          *o       = GEGL_PROPERTIES (operation);
  gfloat                  *in_buf, *out_buf, *out_pixel;
  const Babl              *space   = gegl_operation_get_format (operation, "output");
  gint                     x, y;
  GeglRectangle            src_rect;
  GeglRectangle           *whole_region;
  gdouble                  angle;
  gdouble                  center_x, center_y;

  whole_region = gegl_operation_source_get_bounding_box (operation, "input");

  center_x = gegl_coordinate_relative_to_pixel (
                    o->center_x, whole_region->width);
  center_y = gegl_coordinate_relative_to_pixel (
                    o->center_y, whole_region->height);

  angle = o->angle * G_PI / 180.0;
  while (angle < 0.0)
    angle += 2 * G_PI;

  if (gegl_operation_use_opencl (operation))
    if (cl_process (operation, input, output, roi, whole_region, angle, center_x, center_y))
      return TRUE;

  src_rect = get_required_for_output (operation, "input", roi);

  in_buf    = g_new  (gfloat, src_rect.width * src_rect.height * 4);
  out_buf   = g_new0 (gfloat, roi->width * roi->height * 4);
  out_pixel = out_buf;

  gegl_buffer_get (input, &src_rect, 1.0, babl_format_with_space ("RaGaBaA float", space),
                   in_buf, GEGL_AUTO_ROWSTRIDE, GEGL_ABYSS_NONE);

  for (y = roi->y; y < roi->height + roi->y; ++y)
    {
      for (x = roi->x; x < roi->width + roi->x; ++x)
        {
          gint c, i;
          gfloat sum[] = {0, 0, 0, 0};
          gint count = 0;

          gdouble xr = x - center_x;
          gdouble yr = y - center_y;
          gdouble radius  = sqrt (SQR (xr) + SQR (yr));

          /* This is not the "real" length, a bit shorter */
          gdouble arc_length = radius * angle * SQRT_2;

          /* ensure quality with small angles */
          gint n = MAX (ceil (arc_length), 3);

          /* performance concern */
          if (n > NOMINAL_NUM_IT)
            n = NOMINAL_NUM_IT + (gint) sqrt (n - NOMINAL_NUM_IT);

          /* Iterate over the arc, rotating the offset of the pixel from
           * the center by theta in [angle / 2, -angle / 2).  Computing
           * the sample as the pixel plus a small displacement keeps it
           * precise, and makes the middle sample (theta == 0) land
           * exactly on the pixel itself.
           */
          for (i = 0; i < n; i++)
            {
              gdouble theta = angle * (n - 2 * i) / (2.0 * n);
              gdouble s_val = sin (theta);
              gdouble h_val = sin (theta / 2.0);
              gdouble c_m1  = -2.0 * h_val * h_val; /* cos (theta) - 1 */

              gfloat ix = x + (xr * c_m1 - yr * s_val);
              gfloat iy = y + (xr * s_val + yr * c_m1);

              if (ix >= whole_region->x && ix < whole_region->x + whole_region->width &&
                  iy >= whole_region->y && iy < whole_region->y + whole_region->height)
                {
                  /* do bilinear interpolation to get a nice smooth result */
                  gint   x0 = (gint) floorf (ix);
                  gint   y0 = (gint) floorf (iy);
                  gfloat dx = ix - x0;
                  gfloat dy = iy - y0;

                  gfloat *pix0, *pix1, *pix2, *pix3;
                  gfloat mixy0[4];
                  gfloat mixy1[4];

                  pix0 = get_pixel_color (in_buf, &src_rect, x0,     y0);
                  pix1 = get_pixel_color (in_buf, &src_rect, x0 + 1, y0);
                  pix2 = get_pixel_color (in_buf, &src_rect, x0,     y0 + 1);
                  pix3 = get_pixel_color (in_buf, &src_rect, x0 + 1, y0 + 1);

                  for (c = 0; c < 4; ++c)
                    {
                      mixy0[c] = dy * (pix2[c] - pix0[c]) + pix0[c];
                      mixy1[c] = dy * (pix3[c] - pix1[c]) + pix1[c];

                      sum[c] +=  dx * (mixy1[c] - mixy0[c]) + mixy0[c];
                    }

                  count++;
                }
            }

          if (count == 0)
            {
              gfloat *pix = get_pixel_color (in_buf, &src_rect, x, y);
              for (c = 0; c < 4; ++c)
                *out_pixel++ = pix[c];
            }
          else
            {
              for (c = 0; c < 4; ++c)
                *out_pixel++ = sum[c] / (gfloat) count;
            }
        }
    }

  gegl_buffer_set (output, roi, 0, babl_format_with_space ("RaGaBaA float", space),
                   out_buf, GEGL_AUTO_ROWSTRIDE);

  g_free (in_buf);
  g_free (out_buf);

  return  TRUE;
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

  operation_class->prepare = prepare;
  operation_class->process = operation_process;
  operation_class->get_required_for_output   = get_required_for_output;
  operation_class->get_invalidated_by_change = get_invalidated_by_change;
  operation_class->opencl_support = TRUE;

  filter_class->process    = process;

  gegl_operation_class_set_keys (operation_class,
      "name",               "gegl:motion-blur-circular",
      "title",              _("Circular Motion Blur"),
      "categories",         "blur",
      "position-dependent", "true",
      "license",            "GPL3+",
      "reference-hash",     "540193ad28e012554ef906c12193d450",
      "description", _("Circular motion blur"),
      NULL);
}

#endif
