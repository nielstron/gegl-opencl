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

/* The sample positions are very sensitive to rounding: where the map
 * shrinks the image a lot, the Jacobian entries get large and a change
 * of the coordinates in the last bits of a float moves NoHalo's anchor
 * pixel or the EWA footprint.  Hence the host evaluates the map and
 * the EWA setup in double precision exactly like the CPU path does, and
 * the kernel only does the sampling.  For each output pixel the
 * geometry buffer holds WP_GEO_INTS ints:
 *
 *   ix_0, iy_0           anchor pixel (clamped to near the abyss)
 *   x_0, y_0             (float bits) position relative to the anchor
 *   bounds               EWA box offsets, each + 64 in one byte, 0 if
 *                        LBB-Nohalo is not blended with EWA
 *   theta                (float bits) EWA blend factor
 *   c_major_x, c_major_y, c_minor_x, c_minor_y  (float bits)
 */
#define WP_GEO_INTS     10
#define WP_BAND_PIXELS  (1 << 17)
#define WP_LBB_REACH    2
#define WP_EWA_REACH    13  /* NOHALO_OFFSET_0 in gegl-sampler-nohalo.c */

typedef struct
{
  gdouble       cen_x;
  gdouble       cen_y;
  gdouble       scale_x;
  gdouble       scale_y;
  gdouble       whirl;     /* in radians */
  gdouble       pinch;
  gdouble       radius;
  GeglRectangle abyss;
  GeglRectangle rect;      /* output pixels of the current band */
  gint         *geo;       /* WP_GEO_INTS per pixel of rect */
  gint         *row_box;   /* per row of rect: x0, x1, y0, y1 of the
                            * input pixels the row reads */
} WpGeometry;

static inline gint
wp_float_bits (gfloat f)
{
  union { gfloat f; gint i; } u;

  u.f = f;

  return u.i;
}

/* int_floorf () and int_ceilf () of gegl-buffer-formats.h */
static inline gint
wp_int_floorf (gfloat x)
{
  gint i = (gint) x;

  return i - (i > x);
}

static inline gint
wp_int_ceilf (gfloat x)
{
  return -wp_int_floorf (-x);
}

/* The part of gegl_sampler_get () and gegl_sampler_nohalo_get () that
 * depends on the position and the inverse Jacobian (a b; c d) only,
 * with the same expressions and precision.
 */
static void
wp_pixel_geometry (const WpGeometry *g,
                   gdouble           absolute_x,
                   gdouble           absolute_y,
                   const gdouble     a,
                   const gdouble     b,
                   const gdouble     c,
                   const gdouble     d,
                   gint             *geo,
                   gint             *box)
{
  gdouble fx, fy;
  gfloat  x_0, y_0;
  gint    ix_0, iy_0;
  gint    reach = WP_LBB_REACH;

  if (! isfinite (absolute_x))
    absolute_x = 0.0;
  if (! isfinite (absolute_y))
    absolute_y = 0.0;

  fx = floor (absolute_x);
  fy = floor (absolute_y);

  x_0 = (absolute_x - (gdouble) 0.5) - fx;
  y_0 = (absolute_y - (gdouble) 0.5) - fy;

  /* All reads of anchors further outside are clamped to the edge */
  ix_0 = CLAMP (fx, g->abyss.x - WP_EWA_REACH - 3,
                    g->abyss.x + g->abyss.width + WP_EWA_REACH + 2);
  iy_0 = CLAMP (fy, g->abyss.y - WP_EWA_REACH - 3,
                    g->abyss.y + g->abyss.height + WP_EWA_REACH + 2);

  geo[0] = ix_0;
  geo[1] = iy_0;
  geo[2] = wp_float_bits (x_0);
  geo[3] = wp_float_bits (y_0);
  geo[4] = 0;

  {
    const gdouble aa = a * a;
    const gdouble bb = b * b;
    const gdouble cc = c * c;
    const gdouble dd = d * d;
    const gdouble n11 = aa + bb;
    const gdouble n12 = a * c + b * d;
    const gdouble n21 = n12;
    const gdouble n22 = cc + dd;
    const double det = a * d - b * c;
    const double twice_det = det + det;
    const double frobenius_squared = n11 + n22;
    const double discriminant =
      ( frobenius_squared + twice_det ) * ( frobenius_squared - twice_det );
    const double sqrt_discriminant =
      discriminant > 0. ? sqrt (discriminant) : 0.;
    const gdouble twice_s1s1 = frobenius_squared + sqrt_discriminant;

    if (twice_s1s1 > (gdouble) 2.)
      {
        const gdouble s1s1 = (gdouble) 0.5 * twice_s1s1;
        const gdouble s2s2 = 0.5 * ( frobenius_squared - sqrt_discriminant );
        const gdouble s1s1minusn11 = s1s1 - n11;
        const gdouble s1s1minusn22 = s1s1 - n22;
        const gdouble s1s1minusn11_squared = s1s1minusn11 * s1s1minusn11;
        const gdouble s1s1minusn22_squared = s1s1minusn22 * s1s1minusn22;
        const gdouble temp_u11 =
          s1s1minusn11_squared >= s1s1minusn22_squared ? n12 : s1s1minusn22;
        const gdouble temp_u21 =
          s1s1minusn11_squared >= s1s1minusn22_squared ? s1s1minusn11 : n21;
        const gdouble norm =
          sqrt( temp_u11 * temp_u11 + temp_u21 * temp_u21 );
        const gdouble u11 =
          norm > (gdouble) 0.0 ? temp_u11 / norm : (gdouble) 1.0;
        const gdouble u21 =
          norm > (gdouble) 0.0 ? temp_u21 / norm : (gdouble) 0.0;
        const gdouble major_mag =
          s1s1 <= (gdouble) 1.0 ? (gdouble) 1.0 : sqrt( s1s1 );
        const gdouble minor_mag =
          s2s2 <= (gdouble) 1.0 ? (gdouble) 1.0 : sqrt( s2s2 );
        const gdouble major_unit_x =  u11;
        const gdouble major_unit_y =  u21;
        const gdouble minor_unit_x = -u21;
        const gdouble minor_unit_y =  u11;
        const gfloat c_major_x = major_unit_x / major_mag;
        const gfloat c_major_y = major_unit_y / major_mag;
        const gfloat c_minor_x = minor_unit_x / minor_mag;
        const gfloat c_minor_y = minor_unit_y / minor_mag;
        const gdouble major_x = major_mag * major_unit_x;
        const gdouble major_y = major_mag * major_unit_y;
        const gdouble minor_x = minor_mag * minor_unit_x;
        const gdouble minor_y = minor_mag * minor_unit_y;
        const gdouble ellipse_a =
          major_y * major_y + minor_y * minor_y;
        const gdouble folded_ellipse_b =
          major_x * major_y + minor_x * minor_y;
        const gdouble ellipse_c =
          major_x * major_x + minor_x * minor_x;
        const gdouble ellipse_f = major_mag * minor_mag;
        const gdouble bounding_box_factor =
          ellipse_f * ellipse_f /
          ( ellipse_c * ellipse_a - folded_ellipse_b * folded_ellipse_b );
        const gfloat bounding_box_half_width =
          sqrtf( (gfloat) (ellipse_c * bounding_box_factor) );
        const gfloat bounding_box_half_height =
          sqrtf( (gfloat) (ellipse_a * bounding_box_factor) );
        const gfloat theta = (gfloat) ( (gdouble) 1. / ellipse_f );
        const gint out_left_0 =
          MAX (wp_int_ceilf  ((float) (x_0 - bounding_box_half_width)),
               -WP_EWA_REACH);
        const gint out_rite_0 =
          MIN (wp_int_floorf ((float) (x_0 + bounding_box_half_width)),
               WP_EWA_REACH);
        const gint out_top_0 =
          MAX (wp_int_ceilf  ((float) (y_0 - bounding_box_half_height)),
               -WP_EWA_REACH);
        const gint out_bot_0 =
          MIN (wp_int_floorf ((float) (y_0 + bounding_box_half_height)),
               WP_EWA_REACH);

        /* The sampler loops with do-while, i.e. it visits out_left_0
         * and out_top_0 even if the box is empty.
         */
        geo[4] = (CLAMP (out_left_0, -WP_EWA_REACH, WP_EWA_REACH) + 64)       |
                 (CLAMP (out_rite_0, -WP_EWA_REACH, WP_EWA_REACH) + 64) << 8  |
                 (CLAMP (out_top_0,  -WP_EWA_REACH, WP_EWA_REACH) + 64) << 16 |
                 (CLAMP (out_bot_0,  -WP_EWA_REACH, WP_EWA_REACH) + 64) << 24;
        geo[5] = wp_float_bits (theta);
        geo[6] = wp_float_bits (c_major_x);
        geo[7] = wp_float_bits (c_major_y);
        geo[8] = wp_float_bits (c_minor_x);
        geo[9] = wp_float_bits (c_minor_y);

        reach = WP_EWA_REACH;
      }
  }

  box[0] = MIN (box[0], ix_0 - reach);
  box[1] = MAX (box[1], ix_0 + reach);
  box[2] = MIN (box[2], iy_0 - reach);
  box[3] = MAX (box[3], iy_0 + reach);
}

static inline void
wp_unmap (const WpGeometry *g,
          gdouble           x,
          gdouble           y,
          gdouble          *ux,
          gdouble          *uy)
{
  calc_undistorted_coords (x, y, g->cen_x, g->cen_y, g->scale_x, g->scale_y,
                           g->whirl, g->pinch, g->radius, ux, uy);
}

/* Computes the geometry of rows [offset, offset + size) of g->rect.  The
 * inverse Jacobian is gegl_sampler_compute_scale (), which unmaps the
 * points half a pixel left, right, above and below each pixel and
 * rounds them to float; each of those points is shared by two
 * neighbouring pixels, so they are unmapped only once here.
 */
static void
wp_geometry_rows (gsize    offset,
                  gsize    size,
                  gpointer user_data)
{
  const WpGeometry *g = user_data;
  const gint        w = g->rect.width;
  gfloat           *hor  = g_new (gfloat, 2 * (w + 1));
  gfloat           *vert = g_new (gfloat, 4 * w);
  gfloat           *top  = vert;
  gfloat           *bot  = vert + 2 * w;
  gdouble           ux, uy;
  gint              row, col;

  for (col = 0; col < w; col++)
    {
      wp_unmap (g, g->rect.x + col, g->rect.y + (gint) offset - 0.5, &ux, &uy);
      top[2 * col]     = ux;
      top[2 * col + 1] = uy;
    }

  for (row = offset; row < offset + size; row++)
    {
      const gint  y   = g->rect.y + row;
      gint       *box = g->row_box + 4 * row;
      gfloat     *tmp;

      box[0] = box[2] = G_MAXINT;
      box[1] = box[3] = G_MININT;

      for (col = 0; col < w; col++)
        {
          wp_unmap (g, g->rect.x + col, y + 0.5, &ux, &uy);
          bot[2 * col]     = ux;
          bot[2 * col + 1] = uy;
        }

      for (col = 0; col <= w; col++)
        {
          wp_unmap (g, g->rect.x + col - 0.5, y, &ux, &uy);
          hor[2 * col]     = ux;
          hor[2 * col + 1] = uy;
        }

      for (col = 0; col < w; col++)
        {
          const gfloat a = hor[2 * col + 2] - hor[2 * col];
          const gfloat c = hor[2 * col + 3] - hor[2 * col + 1];
          const gfloat b = bot[2 * col]     - top[2 * col];
          const gfloat d = bot[2 * col + 1] - top[2 * col + 1];

          wp_unmap (g, g->rect.x + col, y, &ux, &uy);

          wp_pixel_geometry (g, ux, uy, a, b, c, d,
                             g->geo + ((gsize) row * w + col) * WP_GEO_INTS,
                             box);
        }

      tmp = top;
      top = bot;
      bot = tmp;
    }

  g_free (hor);
  g_free (vert);
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

/* Implements apply_whirl_pinch () with the NoHalo sampler on the GPU.  The
 * output is processed in bands of rows; for each band the host computes
 * the sampling geometry, and only the part of the input the band reads
 * (clamped to the abyss rectangle, as GEGL_ABYSS_CLAMP does) is uploaded.
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
  static cl_ulong       max_alloc = 0;
  const Babl           *format = gegl_operation_get_format (operation, "output");
  const GeglRectangle  *abyss  = gegl_buffer_get_abyss (input);
  const gint            bpp    = babl_format_get_bytes_per_pixel (format);
  GeglBufferClIterator *i;
  WpGeometry            g;
  cl_mem                in_tex  = NULL;
  cl_mem                geo_tex = NULL;
  gint                 *row_box = NULL;
  gint                 *geo_host;
  cl_int                cl_err  = 0;
  gboolean              err;
  cl_int4               abyss4;
  cl_int                policy  = GEGL_ABYSS_CLAMP;

  if (abyss->width < 1 || abyss->height < 1)
    return FALSE;

  if (!cl_data)
    {
      const char *kernel_name[] = {"whirl_pinch", NULL};
      cl_data = gegl_cl_compile_and_build (whirl_pinch_cl_source, kernel_name);
    }

  if (!cl_data)
    return FALSE;

  if (!max_alloc)
    {
      cl_err = gegl_clGetDeviceInfo (gegl_cl_get_device (),
                                     CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                                     sizeof (cl_ulong), &max_alloc, NULL);
      if (cl_err != CL_SUCCESS || !max_alloc)
        return FALSE;
    }

  /* The same parameters as apply_whirl_pinch () gets from process () */
  g.cen_x   = boundary->width  / 2.0;
  g.cen_y   = boundary->height / 2.0;
  g.scale_x = 1.0;
  g.scale_y = (gdouble) boundary->width / boundary->height;
  g.whirl   = whirl * G_PI / 180;
  g.pinch   = pinch;
  g.radius  = radius;
  g.abyss   = *abyss;

  abyss4.s[0] = abyss->x;
  abyss4.s[1] = abyss->y;
  abyss4.s[2] = abyss->width;
  abyss4.s[3] = abyss->height;

  geo_tex = gegl_clCreateBuffer (gegl_cl_get_context (),
                                 CL_MEM_READ_ONLY,
                                 (size_t) WP_BAND_PIXELS * WP_GEO_INTS *
                                 sizeof (cl_int),
                                 NULL, &cl_err);
  if (cl_err != CL_SUCCESS)
    return FALSE;

  geo_host = g_new (gint, (gsize) WP_BAND_PIXELS * WP_GEO_INTS);

  i = gegl_buffer_cl_iterator_new (output, roi, format, GEGL_CL_BUFFER_WRITE);

  while (gegl_buffer_cl_iterator_next (i, &err))
    {
      const gint w         = i->roi[0].width;
      const gint band_rows = MAX (1, WP_BAND_PIXELS / MAX (w, 1));
      gint       band_y;

      if (err)
        goto error_iter;

      if (w > WP_BAND_PIXELS)
        goto error;

      g_free (row_box);
      row_box = g_new (gint, 4 * band_rows);

      for (band_y = 0; band_y < i->roi[0].height; band_y += band_rows)
        {
          GeglRectangle src_rect;
          cl_int4       srect;
          cl_int        dst_offset = band_y * w;
          size_t        geo_size;
          size_t        global_ws[2];
          gint          x0 = G_MAXINT, x1 = G_MININT;
          gint          y0 = G_MAXINT, y1 = G_MININT;
          gint          row;

          g.rect.x      = i->roi[0].x;
          g.rect.y      = i->roi[0].y + band_y;
          g.rect.width  = w;
          g.rect.height = MIN (band_rows, i->roi[0].height - band_y);
          g.row_box     = row_box;
          geo_size      = (size_t) g.rect.width * g.rect.height *
                          WP_GEO_INTS * sizeof (cl_int);

          /* The host computes this band while the GPU runs the last one */
          g.geo = geo_host;

          gegl_parallel_distribute_range (g.rect.height, 256.0 / w,
                                          wp_geometry_rows, &g);

          for (row = 0; row < g.rect.height; row++)
            {
              x0 = MIN (x0, row_box[4 * row]);
              x1 = MAX (x1, row_box[4 * row + 1]);
              y0 = MIN (y0, row_box[4 * row + 2]);
              y1 = MAX (y1, row_box[4 * row + 3]);
            }

          /* GEGL_ABYSS_CLAMP: only pixels inside the abyss are read */
          x0 = CLAMP (x0, abyss->x, abyss->x + abyss->width  - 1);
          x1 = CLAMP (x1, abyss->x, abyss->x + abyss->width  - 1);
          y0 = CLAMP (y0, abyss->y, abyss->y + abyss->height - 1);
          y1 = CLAMP (y1, abyss->y, abyss->y + abyss->height - 1);

          src_rect.x      = x0;
          src_rect.y      = y0;
          src_rect.width  = x1 - x0 + 1;
          src_rect.height = y1 - y0 + 1;

          if ((cl_ulong) src_rect.width * src_rect.height * bpp > max_alloc)
            goto error;

          srect.s[0] = src_rect.x;
          srect.s[1] = src_rect.y;
          srect.s[2] = src_rect.width;
          srect.s[3] = src_rect.height;

          in_tex = cl_upload_rect (input, &src_rect, format);
          if (!in_tex)
            goto error;

          cl_err = gegl_clEnqueueWriteBuffer (gegl_cl_get_command_queue (),
                                              geo_tex, CL_TRUE, 0,
                                              geo_size, g.geo,
                                              0, NULL, NULL);
          CL_CHECK;

          global_ws[0] = g.rect.width;
          global_ws[1] = g.rect.height;

          cl_err = gegl_cl_set_kernel_args (cl_data->kernel[0],
                                            sizeof (cl_mem),  &in_tex,
                                            sizeof (cl_int4), &srect,
                                            sizeof (cl_int4), &abyss4,
                                            sizeof (cl_int),  &policy,
                                            sizeof (cl_mem),  &geo_tex,
                                            sizeof (cl_mem),  &i->tex[0],
                                            sizeof (cl_int),  &dst_offset,
                                            NULL);
          CL_CHECK;

          cl_err = gegl_clEnqueueNDRangeKernel (gegl_cl_get_command_queue (),
                                                cl_data->kernel[0], 2,
                                                NULL, global_ws, NULL,
                                                0, NULL, NULL);
          CL_CHECK;

          /* The kernel keeps its own reference until it has run */
          gegl_clReleaseMemObject (in_tex);
          in_tex = NULL;
        }
    }

  g_free (row_box);
  g_free (geo_host);
  gegl_clReleaseMemObject (geo_tex);

  return TRUE;

error:
  gegl_buffer_cl_iterator_stop (i);
error_iter:
  if (in_tex)
    gegl_clReleaseMemObject (in_tex);
  g_free (row_box);
  g_free (geo_host);
  gegl_clReleaseMemObject (geo_tex);

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
