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
 * Polarize plug-in --- maps a rectangle to a circle or vice-versa
 * Copyright (C) 1997 Daniel Dunbar
 * Email: ddunbar@diads.com
 * WWW:   http://millennium.diads.com/gimp/
 * Copyright (C) 1997 Federico Mena Quintero
 * federico@nuclecu.unam.mx
 * Copyright (C) 1996 Marc Bless
 * E-mail: bless@ai-lab.fh-furtwangen.de
 * WWW:    www.ai-lab.fh-furtwangen.de/~bless
 *
 * Copyright (C) 2011 Robert Sasu <sasu.robert@gmail.com>
 */

#include "config.h"
#include <glib/gi18n-lib.h>

#ifdef GEGL_PROPERTIES

property_double (depth, _("Circle depth in percent"), 100.0)
  value_range (0.0, 100.0)
  ui_meta     ("unit", "percent")

property_double (angle, _("Offset angle"), 0.0)
  value_range   (0.0, 360.0)
  ui_meta       ("unit", "degree")
  ui_meta       ("direction", "ccw")

property_boolean (bw, _("Map backwards"), FALSE)
  description    (_("Start from the right instead of the left"))

property_boolean (top, _("Map from top"), TRUE)
  description    (_("Put the top row in the middle and the bottom row on the outside"))

property_boolean (polar, _("To polar"), TRUE)
  description    (_("Map the image to a circle"))

property_int  (pole_x, _("X"), 0)
  description (_("Origin point for the polar coordinates"))
  value_range (0, G_MAXINT)
  ui_meta     ("unit", "pixel-coordinate")
  ui_meta     ("axis", "x")
  ui_meta     ("sensitive", "$middle.sensitive & ! middle")

property_int  (pole_y, _("Y"), 0)
  description (_("Origin point for the polar coordinates"))
  value_range (0, G_MAXINT)
  ui_meta     ("unit", "pixel-coordinate")
  ui_meta     ("axis", "y")
  ui_meta     ("sensitive", "$pole-x.sensitive")

property_boolean (middle, _("Choose middle"), TRUE)
  description (_("Let origin point to be the middle one"))
  ui_meta     ("sensitive", "polar")

#else

#define GEGL_OP_FILTER
#define GEGL_OP_NAME     polar_coordinates
#define GEGL_OP_C_SOURCE polar-coordinates.c

#include "gegl-op.h"
#include <stdio.h>

#define WITHIN(a, b, c) ((((a) <= (b)) && ((b) <= (c))) ? 1 : 0)
#define SQR(x) (x)*(x)

#define SCALE_WIDTH     200
#define ENTRY_WIDTH      60

static gboolean
calc_undistorted_coords (gdouble        wx,
                         gdouble        wy,
                         gdouble        cen_x,
                         gdouble        cen_y,
                         gdouble       *x,
                         gdouble       *y,
                         GeglProperties    *o,
                         GeglRectangle  boundary)
{
  gboolean inside;
  gdouble  phi, phi2;
  gdouble  xx, xm, ym, yy;
  gint     xdiff, ydiff;
  gdouble  r;
  gdouble  m;
  gdouble  xmax, ymax, rmax;
  gdouble  x_calc, y_calc;
  gdouble  xi, yi;
  gdouble  circle, angl, t, angle;
  gint     x1, x2, y1, y2;

  /* initialize */

  phi = 0.0;
  r   = 0.0;

  x1     = 0;
  y1     = 0;
  x2     = boundary.width;
  y2     = boundary.height;
  xdiff  = x2 - x1;
  ydiff  = y2 - y1;
  xm     = xdiff / 2.0;
  ym     = ydiff / 2.0;
  circle = o->depth;
  angle  = o->angle;
  angl   = (gdouble) angle / 180.0 * G_PI;

  if (o->polar)
    {
      if (wx >= cen_x)
        {
          if (wy > cen_y)
            {
              phi = G_PI - atan (((double)(wx - cen_x))/
                                 ((double)(wy - cen_y)));
            }
          else if (wy < cen_y)
            {
              phi = atan (((double)(wx - cen_x))/((double)(cen_y - wy)));
            }
          else
            {
              phi = G_PI / 2;
            }
        }
      else if (wx < cen_x)
        {
          if (wy < cen_y)
            {
              phi = 2 * G_PI - atan (((double)(cen_x -wx)) /
                                     ((double)(cen_y - wy)));
            }
          else if (wy > cen_y)
            {
              phi = G_PI + atan (((double)(cen_x - wx))/
                                 ((double)(wy - cen_y)));
            }
          else
            {
              phi = 1.5 * G_PI;
            }
        }

      r   = sqrt (SQR (wx - cen_x) + SQR (wy - cen_y));

      if (wx != cen_x)
        {
          m = fabs (((double)(wy - cen_y)) / ((double)(wx - cen_x)));
        }
      else
        {
          m = 0;
        }

      if (m <= ((double)(y2 - y1) / (double)(x2 - x1)))
        {
          if (wx == cen_x)
            {
              xmax = 0;
              ymax = cen_y - y1;
            }
          else
            {
              xmax = cen_x - x1;
              ymax = m * xmax;
            }
        }
      else
        {
          ymax = cen_y - y1;
          xmax = ymax / m;
        }

      rmax = sqrt ( (double)(SQR (xmax) + SQR (ymax)) );

      t = ((cen_y - y1) < (cen_x - x1)) ? (cen_y - y1) : (cen_x - x1);
      rmax = (rmax - t) / 100 * (100 - circle) + t;

      phi = fmod (phi + angl, 2*G_PI);

      if (o->bw)
        x_calc = x2 - 1 - (x2 - x1 - 1)/(2*G_PI) * phi;
      else
        x_calc = (x2 - x1 - 1)/(2*G_PI) * phi + x1;

      if (o->top)
        y_calc = (y2 - y1)/rmax   * r   + y1;
      else
        y_calc = y2 - (y2 - y1)/rmax * r;
    }
  else
    {
      if (o->bw)
        phi = (2 * G_PI) * (x2 - wx) / xdiff;
      else
        phi = (2 * G_PI) * (wx - x1) / xdiff;

      phi = fmod (phi + angl, 2 * G_PI);

      if (phi >= 1.5 * G_PI)
        phi2 = 2 * G_PI - phi;
      else if (phi >= G_PI)
        phi2 = phi - G_PI;
      else if (phi >= 0.5 * G_PI)
        phi2 = G_PI - phi;
      else
        phi2 = phi;

      xx = tan (phi2);
      if (xx != 0)
        m = (double) 1.0 / xx;
      else
        m = 0;

      if (m <= ((double)(ydiff) / (double)(xdiff)))
        {
          if (phi2 == 0)
            {
              xmax = 0;
              ymax = ym - y1;
            }
          else
            {
              xmax = xm - x1;
              ymax = m * xmax;
            }
        }
      else
        {
          ymax = ym - y1;
          xmax = ymax / m;
        }

      rmax = sqrt ((double)(SQR (xmax) + SQR (ymax)));

      t = ((ym - y1) < (xm - x1)) ? (ym - y1) : (xm - x1);

      rmax = (rmax - t) / 100.0 * (100 - circle) + t;

      if (o->top)
        r = rmax * (double)((wy - y1) / (double)(ydiff));
      else
        r = rmax * (double)((y2 - wy) / (double)(ydiff));

      xx = r * sin (phi2);
      yy = r * cos (phi2);

      if (phi >= 1.5 * G_PI)
        {
          x_calc = (double)xm - xx;
          y_calc = (double)ym - yy;
        }
      else if (phi >= G_PI)
        {
          x_calc = (double)xm - xx;
          y_calc = (double)ym + yy;
        }
      else if (phi >= 0.5 * G_PI)
        {
          x_calc = (double)xm + xx;
          y_calc = (double)ym + yy;
        }
      else
        {
          x_calc = (double)xm + xx;
          y_calc = (double)ym - yy;
        }
    }

  xi = (int) (x_calc + 0.5);
  yi = (int) (y_calc + 0.5);

  inside = (WITHIN (0, xi, boundary.width - 1) && WITHIN (0, yi, boundary.height - 1));
  if (inside)
    {
      *x = x_calc;
      *y = y_calc;
    }
  return inside;
}


static GeglRectangle
get_effective_area (GeglOperation *operation)
{
  GeglRectangle  result = {0,0,0,0};
  GeglRectangle *in_rect = gegl_operation_source_get_bounding_box (operation, "input");

  gegl_rectangle_copy(&result, in_rect);

  return result;
}

#include "opencl/gegl-cl.h"
#include "gegl-buffer-cl-iterator.h"

#include "opencl/polar-coordinates.cl.h"

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

/* In "to polar" = FALSE mode, everything calc_undistorted_coords () computes
 * except the radius depends only on the column, and the result is
 *
 *   x = xm + rmax * sin (phi2) * f,  y = ym + rmax * cos (phi2) * f
 *
 * (with signs depending on the quadrant), f being the row's relative
 * distance from the top or bottom.  The angle is too sensitive for single
 * precision, e.g. 4097 pixels wide its rounding moves the sampled position
 * by up to 0.015 pixels, so the per column factors are computed here in
 * double precision, the same way calc_undistorted_coords () does.
 */
static void
polar_column_factors (GeglProperties      *o,
                      const GeglRectangle *boundary,
                      gint                 wx,
                      gdouble             *fx,
                      gdouble             *fy)
{
  const gint    x1 = 0, y1 = 0;
  const gint    x2 = boundary->width, y2 = boundary->height;
  const gint    xdiff = x2 - x1, ydiff = y2 - y1;
  const gdouble xm = xdiff / 2.0, ym = ydiff / 2.0;
  const gdouble circle = o->depth;
  const gdouble angl = (gdouble) o->angle / 180.0 * G_PI;
  gdouble       phi, phi2, xx, m, xmax, ymax, rmax, t, sx, sy;

  if (o->bw)
    phi = (2 * G_PI) * (x2 - wx) / xdiff;
  else
    phi = (2 * G_PI) * (wx - x1) / xdiff;

  phi = fmod (phi + angl, 2 * G_PI);

  if (phi >= 1.5 * G_PI)
    phi2 = 2 * G_PI - phi;
  else if (phi >= G_PI)
    phi2 = phi - G_PI;
  else if (phi >= 0.5 * G_PI)
    phi2 = G_PI - phi;
  else
    phi2 = phi;

  xx = tan (phi2);
  if (xx != 0)
    m = (double) 1.0 / xx;
  else
    m = 0;

  if (m <= ((double)(ydiff) / (double)(xdiff)))
    {
      if (phi2 == 0)
        {
          xmax = 0;
          ymax = ym - y1;
        }
      else
        {
          xmax = xm - x1;
          ymax = m * xmax;
        }
    }
  else
    {
      ymax = ym - y1;
      xmax = ymax / m;
    }

  rmax = sqrt ((double)(SQR (xmax) + SQR (ymax)));

  t = ((ym - y1) < (xm - x1)) ? (ym - y1) : (xm - x1);

  rmax = (rmax - t) / 100.0 * (100 - circle) + t;

  sx = rmax * sin (phi2);
  sy = rmax * cos (phi2);

  if (phi >= 1.5 * G_PI)
    {
      sx = -sx;
      sy = -sy;
    }
  else if (phi >= G_PI)
    {
      sx = -sx;
    }
  else if (phi < 0.5 * G_PI)
    {
      sy = -sy;
    }

  *fx = sx;
  *fy = sy;
}

/* For a fixed column in "to polar" = FALSE mode, x_calc and y_calc are
 * monotone functions of the row, so the rows calc_undistorted_coords ()
 * reports as inside form one interval.  The decision is too sensitive for
 * single precision near the half-pixel threshold (the margin can be ~1e-5
 * pixels), so the column's first and last inside rows within
 * [y_lo, y_hi] are found here: estimate them from the column factors, then
 * settle them with calc_undistorted_coords () itself.  Returns lo > hi if
 * there are none.
 */
static gboolean
polar_row_inside (GeglProperties      *o,
                  const GeglRectangle *boundary,
                  gdouble              cen_x,
                  gdouble              cen_y,
                  gint                 wx,
                  gint                 wy)
{
  gdouble x, y;

  return calc_undistorted_coords (wx, wy, cen_x, cen_y, &x, &y,
                                  o, *boundary);
}

/* the rows wy for which lo < c0 + c1 * wy < hi, intersected with *a, *b */
static void
polar_limit_rows (gdouble  c0,
                  gdouble  c1,
                  gdouble  lo,
                  gdouble  hi,
                  gdouble *a,
                  gdouble *b)
{
  if (c1 == 0)
    {
      if (! (lo < c0 && c0 < hi))
        {
          *a = 1;
          *b = 0;
        }
    }
  else
    {
      gdouble t0 = (lo - c0) / c1;
      gdouble t1 = (hi - c0) / c1;

      *a = MAX (*a, MIN (t0, t1));
      *b = MIN (*b, MAX (t0, t1));
    }
}

static void
polar_column_rows (GeglProperties      *o,
                   const GeglRectangle *boundary,
                   gdouble              cen_x,
                   gdouble              cen_y,
                   gint                 wx,
                   gdouble              sx,
                   gdouble              sy,
                   gint                 y_lo,
                   gint                 y_hi,
                   cl_int              *row_lo,
                   cl_int              *row_hi)
{
  const gdouble w = boundary->width, h = boundary->height;
  gdouble       r0, r1, a, b;
  gint          lo = 1, hi = 0, k, start;
  gboolean      found_lo = FALSE, found_hi = FALSE;

  /* r = r0 + r1 * wy */
  if (o->top)
    {
      r0 = 0;
      r1 = 1 / h;
    }
  else
    {
      r0 = 1;
      r1 = -1 / h;
    }

  /* (int) (v + 0.5) truncates towards zero, so v is inside for
   * -1.5 < v < size - 0.5
   */
  a = y_lo;
  b = y_hi;
  polar_limit_rows (w / 2 + sx * r0, sx * r1, -1.5, w - 0.5, &a, &b);
  polar_limit_rows (h / 2 + sy * r0, sy * r1, -1.5, h - 0.5, &a, &b);

  if (a <= b + 4)
    {
      start = CLAMP (floor (a) - 2, y_lo, y_hi + 1);
      for (k = start; k <= y_hi && k <= start + 5; k++)
        if (polar_row_inside (o, boundary, cen_x, cen_y, wx, k))
          {
            lo       = k;
            found_lo = TRUE;
            break;
          }

      start = CLAMP (ceil (b) + 2, y_lo - 1, y_hi);
      for (k = start; k >= y_lo && k >= start - 5; k--)
        if (polar_row_inside (o, boundary, cen_x, cen_y, wx, k))
          {
            hi       = k;
            found_hi = TRUE;
            break;
          }
    }

  /* the rows just outside must be outside; otherwise, or if only one end
   * was found, the estimate was off and all rows are checked
   */
  if (found_lo != found_hi ||
      (found_lo &&
       ((lo > y_lo &&
         polar_row_inside (o, boundary, cen_x, cen_y, wx, lo - 1)) ||
        (hi < y_hi &&
         polar_row_inside (o, boundary, cen_x, cen_y, wx, hi + 1)))))
    {
      lo = 1;
      hi = 0;
      found_lo = FALSE;

      for (k = y_lo; k <= y_hi; k++)
        if (polar_row_inside (o, boundary, cen_x, cen_y, wx, k))
          {
            if (! found_lo)
              lo = k;
            found_lo = TRUE;
            hi = k;
          }
    }

  if (! found_lo)
    {
      lo = 1;
      hi = 0;
    }

  *row_lo = lo;
  *row_hi = hi;
}

/* In "to polar" = TRUE mode the kernel's single precision cannot take the
 * hard decisions either: at 3000x2000, top=1, depth=63, pixel (357,562)'s
 * y_calc is 5.5e-5 pixels below the half-pixel threshold, half a single
 * precision ulp, and whether phi + angl wraps around 2 pi moves x_calc by
 * the whole width.  So for each row they are taken here, with
 * calc_undistorted_coords () itself, and uploaded as column intervals.
 *
 * For a fixed row, phi is monotone in the column on either side of the
 * pole, so the wrapped columns of a side are a prefix or a suffix of it.
 * r / rmax, which decides whether a pixel is inside, grows with the
 * distance from the pole column, except that rmax jumps where m crosses
 * (y2 - y1) / (x2 - x1) if the pole is not where that ratio predicts.  m
 * <= (y2 - y1) / (x2 - x1) is exactly |dy| * xdiff <= |dx| * ydiff (the
 * two sides of the double comparison differ by many ulps otherwise), so
 * each side of the row splits into two pieces at |dx| = ceil (|dy| *
 * xdiff / ydiff) in which r / rmax is monotone and the inside pixels form
 * one interval.
 */

/* phi + angl before the wrap and r / rmax, computed the same way as by
 * calc_undistorted_coords (); only used where they are far from any
 * threshold
 */
static void
polar_estimate (GeglProperties      *o,
                const GeglRectangle *boundary,
                gdouble              cen_x,
                gdouble              cen_y,
                gint                 wx,
                gint                 wy,
                gdouble             *v,
                gdouble             *q)
{
  const gdouble w = boundary->width, h = boundary->height;
  gdouble       phi, r, m, xmax, ymax, rmax, t;

  if (wx >= cen_x)
    {
      if (wy > cen_y)
        phi = G_PI - atan ((wx - cen_x) / (wy - cen_y));
      else if (wy < cen_y)
        phi = atan ((wx - cen_x) / (cen_y - wy));
      else
        phi = G_PI / 2;
    }
  else
    {
      if (wy < cen_y)
        phi = 2 * G_PI - atan ((cen_x - wx) / (cen_y - wy));
      else if (wy > cen_y)
        phi = G_PI + atan ((cen_x - wx) / (wy - cen_y));
      else
        phi = 1.5 * G_PI;
    }

  r = sqrt (SQR (wx - cen_x) + SQR (wy - cen_y));
  m = wx != cen_x ? fabs ((wy - cen_y) / (wx - cen_x)) : 0;

  if (m <= h / w)
    {
      xmax = wx == cen_x ? 0 : cen_x;
      ymax = wx == cen_x ? cen_y : m * xmax;
    }
  else
    {
      ymax = cen_y;
      xmax = ymax / m;
    }

  rmax = sqrt (SQR (xmax) + SQR (ymax));
  t    = MIN (cen_x, cen_y);
  rmax = (rmax - t) / 100 * (100 - o->depth) + t;

  *v = phi + o->angle / 180.0 * G_PI;
  *q = r / rmax;
}

typedef struct
{
  GeglProperties      *o;
  const GeglRectangle *boundary;
  gdouble              cen_x;
  gdouble              cen_y;
  gint                 wy;
} PolarRow;

/* 0: outside on the pole's side, 1: inside, 2: outside beyond, mirrored if
 * dir < 0
 */
static gint
polar_rank (const PolarRow *p,
            gint            wx,
            gint            dir)
{
  gdouble x, y, v, q;
  gint    rank = 1;

  if (! calc_undistorted_coords (wx, p->wy, p->cen_x, p->cen_y, &x, &y,
                                 p->o, *p->boundary))
    {
      polar_estimate (p->o, p->boundary, p->cen_x, p->cen_y, wx, p->wy,
                      &v, &q);
      rank = q < 0.5 ? 0 : 2;
    }

  return dir > 0 ? rank : 2 - rank;
}

/* whether calc_undistorted_coords ()'s phi + angl is >= 2 pi */
static gboolean
polar_wraps (const PolarRow *p,
             gint            wx)
{
  const gint w = p->boundary->width;
  gdouble    x, y, v, q;

  polar_estimate (p->o, p->boundary, p->cen_x, p->cen_y, wx, p->wy, &v, &q);

  /* close to the wrap, tell from x_calc, which is then near one end */
  if (fabs (v - 2 * G_PI) < 1e-6 &&
      calc_undistorted_coords (wx, p->wy, p->cen_x, p->cen_y, &x, &y,
                               p->o, *p->boundary))
    return (p->o->bw ? w - 1 - x : x) < (w - 1) / 2.0;

  return v >= 2 * G_PI;
}

/* the first wx in [a, b] with polar_rank () >= level, b + 1 if none */
static gint
polar_search_rank (const PolarRow *p,
                   gint            a,
                   gint            b,
                   gint            dir,
                   gint            level)
{
  b++;

  while (a < b)
    {
      gint mid = a + (b - a) / 2;

      if (polar_rank (p, mid, dir) >= level)
        b = mid;
      else
        a = mid + 1;
    }

  return a;
}

/* the inside columns in [a, b], along which r / rmax is monotone */
static void
polar_piece_inside (const PolarRow *p,
                    gint            a,
                    gint            b,
                    gint            dir,
                    cl_int         *lo,
                    cl_int         *hi)
{
  gint first, last, k;

  *lo = 1;
  *hi = 0;

  if (a > b)
    return;

  first = polar_search_rank (p, a, b, dir, 1);
  last  = polar_search_rank (p, first, b, dir, 2) - 1;

  if (first <= last &&
      (polar_rank (p, first, dir) != 1 || polar_rank (p, last, dir) != 1))
    {
      /* not monotone after all; settle for the hull */
      first = b + 1;
      last  = a - 1;

      for (k = a; k <= b; k++)
        if (polar_rank (p, k, dir) == 1)
          {
            first = MIN (first, k);
            last  = k;
          }
    }

  if (first <= last)
    {
      *lo = first;
      *hi = last;
    }
}

/* the wrapped columns in [a, b], along which phi is monotone, as k and
 * whether they are those >= k (or else those < k)
 */
static void
polar_side_wrap (const PolarRow *p,
                 gint            a,
                 gint            b,
                 cl_int         *k,
                 gint           *up)
{
  gboolean sa, sb;
  gint     lo, hi;

  *k  = a;
  *up = 1;

  if (a > b)
    return;

  sa = polar_wraps (p, a);
  sb = polar_wraps (p, b);

  if (sa == sb)
    {
      *k = sa ? a : b + 1;
      return;
    }

  lo = a;
  hi = b;

  while (lo < hi)
    {
      gint mid = lo + (hi - lo) / 2;

      if (polar_wraps (p, mid) == sb)
        hi = mid;
      else
        lo = mid + 1;
    }

  *k  = lo;
  *up = sb;
}

/* the decisions for row wy and columns [x_lo, x_hi]: s0..s7 the inside
 * intervals, s8 and s9 the wrap columns left and right of the pole, sa
 * whether the wrapped columns are those >= them (bit 0 left, bit 1 right),
 * sb the |dx| from which m <= (y2 - y1) / (x2 - x1)
 */
static void
polar_row_decisions (GeglProperties      *o,
                     const GeglRectangle *boundary,
                     gdouble              cen_x,
                     gdouble              cen_y,
                     gint                 wy,
                     gint                 x_lo,
                     gint                 x_hi,
                     cl_int16            *d)
{
  const gint64 dy  = ABS ((gint64) wy - (gint64) cen_y);
  const gint64 thr = (dy * boundary->width + boundary->height - 1) /
                     boundary->height;
  const gint   cx  = cen_x;
  gint         t, up_l, up_r;
  PolarRow     p   = {o, boundary, cen_x, cen_y, wy};

  /* |dx| >= t on the outer pieces */
  t = MIN (thr, G_MAXINT / 4);

  polar_piece_inside (&p, x_lo, MIN (x_hi, (gint64) cx - MAX (t, 1)), -1,
                      &d->s[0], &d->s[1]);
  polar_piece_inside (&p, MAX (x_lo, (gint64) cx - MAX (t, 1) + 1),
                      MIN (x_hi, cx - 1), -1,
                      &d->s[2], &d->s[3]);
  polar_piece_inside (&p, MAX (x_lo, cx), MIN (x_hi, (gint64) cx + t - 1), 1,
                      &d->s[4], &d->s[5]);
  polar_piece_inside (&p, MAX (x_lo, (gint64) cx + t), x_hi, 1,
                      &d->s[6], &d->s[7]);

  polar_side_wrap (&p, x_lo, MIN (x_hi, cx - 1), &d->s[8], &up_l);
  polar_side_wrap (&p, MAX (x_lo, cx), x_hi, &d->s[9], &up_r);

  d->s[10] = up_l | (up_r << 1);
  d->s[11] = t;
}

/* Implements process () with the NoHalo sampler on the GPU.  Samples are
 * only taken at positions that round to a pixel inside the input's bounding
 * box, so only the part of the abyss rectangle that NoHalo's stencil can
 * reach from there needs to be uploaded; GEGL_ABYSS_NONE handles the rest.
 */
static gboolean
cl_process (GeglOperation       *operation,
            GeglBuffer          *input,
            GeglBuffer          *output,
            const GeglRectangle *roi,
            const GeglRectangle *boundary,
            gdouble              cen_x,
            gdouble              cen_y)
{
  GeglProperties       *o      = GEGL_PROPERTIES (operation);
  const Babl           *format = gegl_operation_get_format (operation, "output");
  const Babl           *space  = babl_format_get_space (format);
  const Babl           *interpolate_format;
  const GeglRectangle  *abyss  = gegl_buffer_get_abyss (input);
  GeglRectangle         src_rect;
  GeglBufferClIterator *i;
  cl_mem                in_tex = NULL;
  cl_mem                col_tex = NULL;
  cl_mem                row_tex = NULL;
  cl_mem                prow_tex = NULL;
  cl_float2            *cols;
  cl_int2              *rows;
  cl_int16             *prows;
  cl_int                n_cols;
  cl_int                n_prows;
  cl_int                cl_err = 0;
  gboolean              err;
  cl_int4               srect;
  cl_int4               abyss4;
  cl_int                policy  = GEGL_ABYSS_NONE;
  cl_float              f_cen_x = cen_x;
  cl_float              f_cen_y = cen_y;
  cl_int                width   = boundary->width;
  cl_int                height  = boundary->height;
  cl_float              circle  = o->depth;
  cl_float              angl    = o->angle / 180.0 * G_PI;
  cl_int                bw      = o->bw;
  cl_int                top     = o->top;
  cl_int                polar   = o->polar;
  cl_int                unpremultiply;

  interpolate_format = babl_format_with_space ("RaGaBaA float", space);

  if (format == interpolate_format)
    unpremultiply = 0;
  else if (format == babl_format_with_space ("RGBA float", space))
    unpremultiply = 1;
  else
    return FALSE;

  /* NoHalo reads up to two pixels around the sampled one */
  src_rect = *GEGL_RECTANGLE (-3, -3, width + 6, height + 6);
  if (! gegl_rectangle_intersect (&src_rect, &src_rect, abyss))
    return FALSE;

  if ((gdouble) src_rect.width * src_rect.height > (1 << 26))
    return FALSE;

  /* the kernel takes the pole's coordinates as exact floats */
  if (cen_x >= (1 << 24) || cen_y >= (1 << 24))
    return FALSE;

  if (!cl_data)
    {
      const char *kernel_name[] = {"polar_coordinates", NULL};
      cl_data = gegl_cl_compile_and_build (polar_coordinates_cl_source,
                                           kernel_name);
    }

  if (!cl_data)
    return FALSE;

  srect.s[0]  = src_rect.x;
  srect.s[1]  = src_rect.y;
  srect.s[2]  = src_rect.width;
  srect.s[3]  = src_rect.height;
  abyss4.s[0] = abyss->x;
  abyss4.s[1] = abyss->y;
  abyss4.s[2] = abyss->width;
  abyss4.s[3] = abyss->height;

  /* per column factors and inside rows for polar == FALSE; a dummy entry
   * otherwise
   */
  n_cols = polar ? 1 : roi->width;
  cols   = g_new0 (cl_float2, n_cols);
  rows   = g_new0 (cl_int2, n_cols);

  if (! polar)
    {
      gint x;

      for (x = 0; x < roi->width; x++)
        {
          gdouble sx, sy;

          polar_column_factors (o, boundary, roi->x + x, &sx, &sy);
          cols[x].s[0] = sx;
          cols[x].s[1] = sy;
          polar_column_rows (o, boundary, cen_x, cen_y, roi->x + x, sx, sy,
                             roi->y, roi->y + roi->height - 1,
                             &rows[x].s[0], &rows[x].s[1]);
        }
    }

  /* per row decisions for polar == TRUE; a dummy entry otherwise */
  n_prows = polar ? roi->height : 1;
  prows   = g_new0 (cl_int16, n_prows);

  if (polar)
    {
      gint y;

      for (y = 0; y < roi->height; y++)
        polar_row_decisions (o, boundary, cen_x, cen_y, roi->y + y,
                             roi->x, roi->x + roi->width - 1, &prows[y]);
    }

  prow_tex = gegl_clCreateBuffer (gegl_cl_get_context (),
                                  CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY,
                                  n_prows * sizeof (cl_int16), prows,
                                  &cl_err);
  g_free (prows);
  if (cl_err != CL_SUCCESS)
    {
      g_free (cols);
      g_free (rows);
      return FALSE;
    }

  col_tex = gegl_clCreateBuffer (gegl_cl_get_context (),
                                 CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY,
                                 n_cols * sizeof (cl_float2), cols, &cl_err);
  g_free (cols);
  if (cl_err != CL_SUCCESS)
    {
      g_free (rows);
      gegl_clReleaseMemObject (prow_tex);
      return FALSE;
    }

  row_tex = gegl_clCreateBuffer (gegl_cl_get_context (),
                                 CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY,
                                 n_cols * sizeof (cl_int2), rows, &cl_err);
  g_free (rows);
  if (cl_err != CL_SUCCESS)
    {
      gegl_clReleaseMemObject (col_tex);
      gegl_clReleaseMemObject (prow_tex);
      return FALSE;
    }

  in_tex = cl_upload_rect (input, &src_rect, interpolate_format);
  if (!in_tex)
    {
      gegl_clReleaseMemObject (col_tex);
      gegl_clReleaseMemObject (row_tex);
      gegl_clReleaseMemObject (prow_tex);
      return FALSE;
    }

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
                                        sizeof (cl_float), &f_cen_x,
                                        sizeof (cl_float), &f_cen_y,
                                        sizeof (cl_int),   &width,
                                        sizeof (cl_int),   &height,
                                        sizeof (cl_float), &circle,
                                        sizeof (cl_float), &angl,
                                        sizeof (cl_int),   &bw,
                                        sizeof (cl_int),   &top,
                                        sizeof (cl_int),   &polar,
                                        sizeof (cl_mem),   &col_tex,
                                        sizeof (cl_mem),   &row_tex,
                                        sizeof (cl_int),   &roi->x,
                                        sizeof (cl_mem),   &prow_tex,
                                        sizeof (cl_int),   &roi->y,
                                        sizeof (cl_int),   &unpremultiply,
                                        NULL);
      CL_CHECK;

      cl_err = gegl_clEnqueueNDRangeKernel (gegl_cl_get_command_queue (),
                                            cl_data->kernel[0], 2,
                                            NULL, global_ws, NULL,
                                            0, NULL, NULL);
      CL_CHECK;
    }

  gegl_clReleaseMemObject (in_tex);
  gegl_clReleaseMemObject (col_tex);
  gegl_clReleaseMemObject (row_tex);
  gegl_clReleaseMemObject (prow_tex);

  return TRUE;

error:
  gegl_buffer_cl_iterator_stop (i);
error_iter:
  gegl_clReleaseMemObject (in_tex);
  gegl_clReleaseMemObject (col_tex);
  gegl_clReleaseMemObject (row_tex);
  gegl_clReleaseMemObject (prow_tex);

  return FALSE;
}

static gboolean
process (GeglOperation       *operation,
         GeglBuffer          *input,
         GeglBuffer          *output,
         const GeglRectangle *result,
         gint                 level)
{
  GeglProperties          *o            = GEGL_PROPERTIES (operation);
  GeglRectangle            boundary     = get_effective_area (operation);
  const Babl              *format       = gegl_operation_get_format (operation, "output");
  GeglSampler             *sampler      = gegl_buffer_sampler_new_at_level (
                                    input, format, GEGL_SAMPLER_NOHALO,
                                    level);

  gint      x,y;
  gfloat   *src_buf, *dst_buf;
  gfloat    dest[4];
  gint      i, offset = 0;
  gboolean  inside;
  gdouble   px, py;
  gdouble   cen_x, cen_y;

  GeglBufferMatrix2  scale;  /* a matrix indicating scaling factors around the
                                current center pixel.
                             */

  if (o->middle)
    {
      cen_x = boundary.width / 2;
      cen_y = boundary.height / 2;
    }
  else
    {
      cen_x = o->pole_x;
      cen_y = o->pole_y;
    }

  if (level == 0 && gegl_operation_use_opencl (operation) &&
      cl_process (operation, input, output, result, &boundary, cen_x, cen_y))
    {
      g_object_unref (sampler);
      return TRUE;
    }

  src_buf = g_new0 (gfloat, result->width * result->height * 4);
  dst_buf = g_new0 (gfloat, result->width * result->height * 4);

  gegl_buffer_get (input, result, 1.0, format, src_buf, GEGL_AUTO_ROWSTRIDE, GEGL_ABYSS_NONE);

  for (y = result->y; y < result->y + result->height; y++)
    for (x = result->x; x < result->x + result->width; x++)
      {
#define gegl_unmap(u,v,ud,vd) {                                         \
          gdouble rx = 0.0, ry = 0.0;                                   \
          inside = calc_undistorted_coords ((gdouble)x, (gdouble)y,     \
                                            cen_x, cen_y, &rx, &ry,     \
                                            o, boundary);               \
          ud = rx;                                                      \
          vd = ry;                                                      \
        }
        gegl_sampler_compute_scale (scale, x, y);
        gegl_unmap(x,y,px,py);
#undef gegl_unmap

        if (inside)
          gegl_sampler_get (sampler, px, py, &scale, dest,
                            GEGL_ABYSS_NONE);
        else
          for (i=0; i<4; i++)
            dest[i] = 0.0;

        for (i=0; i<4; i++)
          dst_buf[offset++] = dest[i];
      }

  gegl_buffer_set (output, result, 0, format, dst_buf, GEGL_AUTO_ROWSTRIDE);

  g_free (src_buf);
  g_free (dst_buf);

  g_object_unref (sampler);

  return  TRUE;
}

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

  operation_class->get_required_for_output = get_required_for_output;
  operation_class->process                 = operation_process;

  filter_class->process                    = process;
  operation_class->opencl_support          = TRUE;

  gegl_operation_class_set_keys (operation_class,
    "name",               "gegl:polar-coordinates",
    "title",              _("Polar Coordinates"),
    "categories",         "transform:map",
    "position-dependent", "true",
    "reference-hash",     "4716987c6105311bd29937d5d427f59b",
    "license",            "GPL3+",
    "description", _("Convert image to or from polar coordinates"),
    NULL);
}

#endif
