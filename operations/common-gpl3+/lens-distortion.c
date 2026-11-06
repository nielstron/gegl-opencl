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
 * Lens plug-in - adjust for lens distortion
 *
 * Copyright (C) 1995 Spencer Kimball and Peter Mattis
 * Copyright (C) 2001-2005 David Hodson <hodsond@acm.org>
 * Copyright (C) 2008 Aurimas Juška <aurisj@svn.gnome.org>
 * Copyright (C) 2011 Robert Sasu <sasu.robert@gmail.com>
 * Copyright (C) 2013 Téo Mazars <teo.mazars@ensimag.fr>
 *
 * Many thanks for Lars Clausen for the original inspiration,
 *   useful discussion, optimisation and improvements.
 */

#include "config.h"
#include <glib/gi18n-lib.h>

#ifdef GEGL_PROPERTIES

property_double (main, _("Main"), 0.0)
    description (_("Amount of second-order distortion"))
    value_range (-100.0, 100.0)

property_double (edge, _("Edge"), 0.0)
    description (_("Amount of fourth-order distortion"))
    value_range (-100.0, 100.0)

property_double (zoom, _("Zoom"), 0.0)
    description (_("Rescale overall image size"))
    value_range (-100, 100.0)

property_double (x_shift, _("Shift X"), 0.0)
    description (_("Effect center offset in X"))
    value_range (-100.0, 100.0)

property_double (y_shift, _("Shift Y"), 0.0)
    description (_("Effect center offset in Y"))
    value_range (-100.0, 100.0)

property_double (brighten, _("Brighten"), 0.0)
    description (_("Adjust brightness in corners"))
    value_range (-100.0, 100.0)

property_color  (background, _("Background color"), "none")

#else

#define GEGL_OP_FILTER
#define GEGL_OP_NAME     lens_distortion
#define GEGL_OP_C_SOURCE lens-distortion.c

#include "gegl-op.h"
#include <stdio.h>

#define SQR(x) ((x)*(x))

#define MIN3(x,y,z) (MIN (MIN ((x),(y)),(z)))

#define MAX3(x,y,z) (MAX (MAX ((x),(y)),(z)))

#define MAX_WH     1024
#define CHUNK_SIZE 512


typedef struct
{
  gdouble centre_x;
  gdouble centre_y;
  gdouble mult_sq;
  gdouble mult_qd;
  gdouble rescale;
  gdouble brighten;
  gdouble norm;
} LensValues;


static void
reorder (gdouble *low,
         gdouble *high)
{
  gdouble temp;

  if (*low < *high) return;

  temp = *low;
  *low = *high;
  *high = temp;
}

static LensValues
lens_setup_calc (GeglProperties *o,
                 GeglRectangle   boundary)
{
  LensValues lens;

  lens.norm     = 4.0 / (SQR (boundary.width) + SQR (boundary.height));
  lens.centre_x = boundary.width  * (100.0 + o->x_shift) / 200.0;
  lens.centre_y = boundary.height * (100.0 + o->y_shift) / 200.0;
  lens.mult_sq  = o->main / 200.0;
  lens.mult_qd  = o->edge / 200.0;
  lens.rescale  = pow (2.0, - o->zoom / 100.0);
  lens.brighten = - o->brighten / 10.0;

  return lens;
}

static void
lens_get_source_coord (gdouble     i,
                       gdouble     j,
                       gdouble    *x,
                       gdouble    *y,
                       gdouble    *mag,
                       LensValues *lens)
{
  gdouble radius_sq, off_x, off_y, radius_mult;

  off_x = i - lens->centre_x;
  off_y = j - lens->centre_y;

  radius_sq = SQR (off_x) + SQR (off_y);

  radius_sq *= lens->norm;

  radius_mult = radius_sq * lens->mult_sq + SQR (radius_sq) * lens->mult_qd;

  *mag = radius_mult;

  radius_mult = lens->rescale * (1.0 + radius_mult);

  *x = lens->centre_x + radius_mult * off_x;
  *y = lens->centre_y + radius_mult * off_y;
}

/* FIXME: not 100% bullet proof */
static GeglRectangle
get_required (GeglRectangle       *boundary,
              const GeglRectangle *roi,
              GeglOperation       *operation)
{

  GeglProperties *o;
  GeglRectangle   area;
  LensValues      lens;
  gdouble         x1, y1, x2, y2, x3, y3, x4, y4, mag;
  gint            x, y, width, height;

  o = GEGL_PROPERTIES (operation);

  lens = lens_setup_calc (o, *boundary);

  x = roi->x;
  y = roi->y;
  width = roi->width;
  height = roi->height;

  lens_get_source_coord (x,         y,          &x1, &y1, &mag, &lens);
  lens_get_source_coord (x + width, y,          &x2, &y2, &mag, &lens);
  lens_get_source_coord (x,         y + height, &x3, &y3, &mag, &lens);
  lens_get_source_coord (x + width, y + height, &x4, &y4, &mag, &lens);

  /* This is ugly, and happens
   * with a crazy set of parameters */
  reorder (&x1, &x2);
  reorder (&x3, &x4);

  reorder (&y1, &y3);
  reorder (&y2, &y4);

  if (lens.centre_y > y && lens.centre_y < y + height)
    {
      gdouble x5, y5, x6, y6;

      lens_get_source_coord (x,         lens.centre_y, &x5, &y5, &mag, &lens);
      lens_get_source_coord (x + width, lens.centre_y, &x6, &y6, &mag, &lens);

      reorder (&x5, &x6);

      area.x = floor (MIN3 (x1, x3, x5)) - 1;
      area.width = ceil (MAX3 (x2, x4, x6)) + 3 - area.x;
    }
  else
    {
      area.x = floor (MIN (x1, x3)) - 1;
      area.width = ceil (MAX (x2, x4)) + 3 - area.x;
    }

  if (lens.centre_x > x && lens.centre_x < x + width)
    {
      gdouble x5, y5, x6, y6;

      lens_get_source_coord (lens.centre_x, y,          &x5, &y5, &mag, &lens);
      lens_get_source_coord (lens.centre_x, y + height, &x6, &y6, &mag, &lens);

      reorder (&y5, &y6);

      area.y = floor (MIN3 (y1, y2, y5)) - 1;
      area.height = ceil (MAX3 (y3, y4, y6)) + 3 - area.y;
    }
  else
    {
      area.y = floor (MIN (y1, y2)) - 1;
      area.height = ceil (MAX (y3, y4)) + 3 - area.y;
    }

  return area;
}

static void
clamp_area (GeglRectangle *area,
            gdouble        center_x,
            gdouble        center_y)
{
  if (center_x <= area->x || area->width < 1)
    {
      area->width = CLAMP (area->width, 1, MAX_WH);
    }
  else
    {
      area->x += area->width - CLAMP (area->width, 1, MAX_WH);
      area->width = CLAMP (area->width, 1, MAX_WH);
    }

  if (center_y <= area->y || area->height < 1)
    {
      area->height = CLAMP (area->height, 1, MAX_WH);
    }
  else
    {
      area->y += area->height - CLAMP (area->height, 1, MAX_WH);
      area->height = CLAMP (area->height, 1, MAX_WH);
    }
}

static GeglRectangle
get_required_for_output (GeglOperation       *operation,
                         const gchar         *input_pad,
                         const GeglRectangle *roi)
{
  GeglRectangle *boundary;

  boundary = gegl_operation_source_get_bounding_box (operation, "input");

  if (strcmp (input_pad, "input") || !boundary)
    return *GEGL_RECTANGLE (0, 0, 0, 0);

  return get_required (boundary, roi, operation);
}

/*
 * Catmull-Rom cubic interpolation XXX: FIXME: use gegl resampler instead of
 *                                             reimplementing cubic sampler here
 * equally spaced points p0, p1, p2, p3
 * interpolate 0 <= u < 1 between p1 and p2
 *
 * (1 u u^2 u^3) (  0.0  1.0  0.0  0.0 ) (p0)
 *               ( -0.5  0.0  0.5  0.0 ) (p1)
 *               (  1.0 -2.5  2.0 -0.5 ) (p2)
 *               ( -0.5  1.5 -1.5  0.5 ) (p3)
 *
 */

static void
lens_cubic_interpolate (gfloat  *src,
                        gfloat  *dst,
                        gdouble  dx,
                        gdouble  dy,
                        gdouble  brighten)
{
  gfloat um1, u, up1, up2;
  gfloat vm1, v, vp1, vp2;
  gint   c, row_stride = 16;
  gfloat verts[16];

  um1 = ((-0.5 * dx + 1.0) * dx - 0.5) * dx;
  u = (1.5 * dx - 2.5) * dx * dx + 1.0;
  up1 = ((-1.5 * dx + 2.0) * dx + 0.5) * dx;
  up2 = (0.5 * dx - 0.5) * dx * dx;

  vm1 = ((-0.5 * dy + 1.0) * dy - 0.5) * dy;
  v = (1.5 * dy - 2.5) * dy * dy + 1.0;
  vp1 = ((-1.5 * dy + 2.0) * dy + 0.5) * dy;
  vp2 = (0.5 * dy - 0.5) * dy * dy;

  for (c = 0; c < 4 * 4; ++c)
    {
      verts[c] = vm1 * src[c] + v * src[c + row_stride] +
        vp1 * src[c + row_stride * 2] + vp2 * src[ c + row_stride * 3];
    }

  for (c = 0; c < 4; ++c)
    {
      gfloat result;

      result = um1 * verts[c] + u * verts[c + 4] +
        up1 * verts[c + 4 * 2] + up2 * verts[c + 4 * 3];

      if (c != 3)
        result *= brighten;

      dst[c] = CLAMP (result, 0.0, 1.0);
    }
}

static void
lens_distort_func (gfloat              *src_buf,
                   gfloat              *dst_buf,
                   const GeglRectangle *extended,
                   const GeglRectangle *result,
                   const GeglRectangle *boundary,
                   LensValues          *lens,
                   gint                 xx,
                   gint                 yy,
                   GeglSampler         *sampler,
                   gfloat              *background,
                   gint                 level)
{
  gdouble sx, sy, mag;
  gdouble brighten;
  gfloat  pixel_buffer [16 * 4], temp[4];
  gdouble dx, dy;
  gint    x_int, y_int, x = 0, y = 0, offset = 0;

  temp[0] = temp[1] = temp[2] = temp[3] = 0.0;

  lens_get_source_coord ((gdouble) xx, (gdouble) yy, &sx, &sy, &mag, lens);

  /* pseudo gamma transformation, since the input is scRGB */
  brighten = pow (MAX (1.0 + mag * lens->brighten, 0.0), 2.4);

  x_int = floor (sx);
  dx = sx - x_int;

  y_int = floor (sy);
  dy = sy - y_int;

  for (y = y_int - 1; y <= y_int + 2; y++)
    {
      for (x = x_int - 1; x <= x_int + 2; x++)
        {
          gint b;

          if (x < boundary->x || x >= (boundary->x + boundary->width) ||
              y < boundary->y || y >= (boundary->y + boundary->height))
            {
              for (b = 0; b < 4; b++)
                pixel_buffer[offset++] = background[b];
            }
          else
            {

              if (x >= extended->x && x < (extended->x + extended->width) &&
                  y >= extended->y && y < (extended->y + extended->height))
                {
                  gint src_off;
                  src_off = (y - extended->y) * extended->width * 4 +
                    (x - extended->x) * 4;

                  for (b = 0; b < 4; b++)
                    temp[b] = src_buf[src_off++];
                }
              else
                {
                  gegl_sampler_get (sampler,
                                    x, y, NULL, temp, GEGL_ABYSS_CLAMP);
                }

              for (b = 0; b < 4; b++)
                pixel_buffer[offset++] = temp[b];
            }
        }
    }

  lens_cubic_interpolate (pixel_buffer, temp, dx, dy, brighten);

  offset = (yy - result->y) * result->width * 4 + (xx - result->x) * 4;

  for (x = 0; x < 4; x++)
    dst_buf[offset++] = temp[x];
}

#include "opencl/gegl-cl.h"
#include "gegl-buffer-cl-iterator.h"

#include "opencl/lens-distortion.cl.h"

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

  return TRUE;
}

/* @value as a pair of floats whose sum is @value with about twice the
 * precision of a float.
 */
static cl_float2
cl_float_float (gdouble value)
{
  cl_float2 ff;

  ff.s[0] = value;
  ff.s[1] = value - (gdouble) ff.s[0];

  return ff;
}

#define BOUNDS_SEG 256

/* Computes the part of @boundary that the samples of the pixels in
 * @result read, with the kernel's own source coordinates.  Pixels whose
 * source pixel lies outside the chunk's area are read with a linear
 * sampler that averages them with their top-left neighbours, so those
 * are included.  Returns FALSE on error.
 */
static gboolean
cl_get_needed (const GeglRectangle *result,
               const GeglRectangle *boundary,
               const cl_float2     *coord_args,
               GeglRectangle       *needed)
{
  cl_int   cl_err = 0;
  cl_mem   bounds_tex = NULL;
  cl_int4 *bounds = NULL;
  cl_int4  limit;
  cl_int   seg    = BOUNDS_SEG;
  size_t   global_ws[2];
  size_t   n, k;
  gint     x0 = G_MAXINT, y0 = G_MAXINT, x1 = G_MININT, y1 = G_MININT;

  /* the samples of source position (x, y) are x - 1 ... x + 2 and
   * y - 1 ... y + 2, so a position outside 'limit' only samples outside
   * @boundary */
  limit.s[0] = boundary->x - 3;
  limit.s[1] = boundary->y - 3;
  limit.s[2] = boundary->x + boundary->width + 1;
  limit.s[3] = boundary->y + boundary->height + 1;

  global_ws[0] = (result->width + BOUNDS_SEG - 1) / BOUNDS_SEG;
  global_ws[1] = result->height;
  n            = global_ws[0] * global_ws[1];

  bounds_tex = gegl_clCreateBuffer (gegl_cl_get_context (),
                                    CL_MEM_WRITE_ONLY,
                                    n * sizeof (cl_int4), NULL, &cl_err);
  CL_CHECK;

  cl_err = gegl_cl_set_kernel_args (cl_data->kernel[1],
                                    sizeof (cl_mem),    &bounds_tex,
                                    sizeof (cl_int),    &result->x,
                                    sizeof (cl_int),    &result->y,
                                    sizeof (cl_int),    &result->width,
                                    sizeof (cl_int),    &seg,
                                    sizeof (cl_int4),   &limit,
                                    sizeof (cl_float2), &coord_args[0],
                                    sizeof (cl_float2), &coord_args[1],
                                    sizeof (cl_float2), &coord_args[2],
                                    sizeof (cl_float2), &coord_args[3],
                                    sizeof (cl_float2), &coord_args[4],
                                    sizeof (cl_float2), &coord_args[5],
                                    NULL);
  CL_CHECK;

  cl_err = gegl_clEnqueueNDRangeKernel (gegl_cl_get_command_queue (),
                                        cl_data->kernel[1], 2,
                                        NULL, global_ws, NULL,
                                        0, NULL, NULL);
  CL_CHECK;

  bounds = g_new (cl_int4, n);

  cl_err = gegl_clEnqueueReadBuffer (gegl_cl_get_command_queue (),
                                     bounds_tex, CL_TRUE, 0,
                                     n * sizeof (cl_int4), bounds,
                                     0, NULL, NULL);
  CL_CHECK;

  for (k = 0; k < n; k++)
    {
      x0 = MIN (x0, bounds[k].s[0]);
      y0 = MIN (y0, bounds[k].s[1]);
      x1 = MAX (x1, bounds[k].s[2]);
      y1 = MAX (y1, bounds[k].s[3]);
    }

  g_free (bounds);
  gegl_clReleaseMemObject (bounds_tex);

  /* the samples inside @boundary */
  x0 = MAX (x0 - 1, boundary->x);
  y0 = MAX (y0 - 1, boundary->y);
  x1 = MIN (x1 + 2, boundary->x + boundary->width  - 1);
  y1 = MIN (y1 + 2, boundary->y + boundary->height - 1);

  if (x0 > x1 || y0 > y1)
    {
      /* every sample is background, nothing is read */
      *needed = *GEGL_RECTANGLE (boundary->x, boundary->y, 1, 1);
      return TRUE;
    }

  /* the linear sampler's top-left neighbours, and a pixel of margin */
  *needed = *GEGL_RECTANGLE (x0 - 2, y0 - 2, x1 - x0 + 4, y1 - y0 + 4);

  return TRUE;

error:
  g_free (bounds);
  if (bounds_tex)
    gegl_clReleaseMemObject (bounds_tex);

  return FALSE;
}

static gboolean
cl_process (GeglOperation       *operation,
            GeglBuffer          *input,
            GeglBuffer          *output,
            const GeglRectangle *result,
            const GeglRectangle *boundary,
            LensValues          *lens,
            const gfloat        *background)
{
  const Babl           *out_format = gegl_operation_get_format (operation, "output");
  const Babl           *format     = babl_format_with_space ("RaGaBaA float",
                                                             out_format);
  const GeglRectangle  *abyss      = gegl_buffer_get_abyss (input);
  GeglRectangle         needed;
  GeglRectangle         src_rect;
  GeglBufferClIterator *i;
  cl_mem                in_tex     = NULL;
  cl_mem                areas_tex  = NULL;
  cl_int                cl_err     = 0;
  gboolean              err;
  cl_int4              *areas;
  cl_int4               srect;
  cl_int4               abyss4;
  cl_int4               boundary4;
  cl_float4             bg;
  cl_int                chunks_x   = (result->width  + CHUNK_SIZE - 1) / CHUNK_SIZE;
  cl_int                chunks_y   = (result->height + CHUNK_SIZE - 1) / CHUNK_SIZE;
  cl_int                unpremultiply;
  cl_float2             coord_args[6];
  cl_float              brighten   = lens->brighten;
  cl_ulong              max_alloc  = 0;
  gint                  j, k;

  if (out_format == format)
    unpremultiply = 0;
  else if (out_format == babl_format_with_space ("RGBA float", out_format))
    unpremultiply = 1;
  else
    return FALSE;

  /* the source area of each chunk, as process () computes it */
  areas = g_new (cl_int4, chunks_x * chunks_y);

  for (k = 0; k < chunks_y; k++)
    for (j = 0; j < chunks_x; j++)
      {
        GeglRectangle chunk;
        GeglRectangle area;

        chunk = *GEGL_RECTANGLE (result->x + j * CHUNK_SIZE,
                                 result->y + k * CHUNK_SIZE,
                                 CHUNK_SIZE, CHUNK_SIZE);
        gegl_rectangle_intersect (&chunk, &chunk, result);

        area = get_required ((GeglRectangle *) boundary, &chunk, operation);
        clamp_area (&area, lens->centre_x, lens->centre_y);

        areas[k * chunks_x + j].s[0] = area.x;
        areas[k * chunks_x + j].s[1] = area.y;
        areas[k * chunks_x + j].s[2] = area.width;
        areas[k * chunks_x + j].s[3] = area.height;
      }

  coord_args[0] = cl_float_float (lens->centre_x);
  coord_args[1] = cl_float_float (lens->centre_y);
  coord_args[2] = cl_float_float (lens->mult_sq);
  coord_args[3] = cl_float_float (lens->mult_qd);
  coord_args[4] = cl_float_float (lens->rescale);
  coord_args[5] = cl_float_float (lens->norm);

  if (!cl_data)
    {
      const char *kernel_name[] = {"lens_distortion",
                                   "lens_distortion_bounds", NULL};
      cl_data = gegl_cl_compile_and_build (lens_distortion_cl_source,
                                           kernel_name);
    }

  if (!cl_data)
    {
      g_free (areas);
      return FALSE;
    }

  /* upload only the part of the input that is read */
  if (! cl_get_needed (result, boundary, coord_args, &needed) ||
      ! cl_get_source_rect (&needed, abyss, GEGL_ABYSS_CLAMP, &src_rect))
    {
      g_free (areas);
      return FALSE;
    }

  cl_err = gegl_clGetDeviceInfo (gegl_cl_get_device (),
                                 CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                                 sizeof (max_alloc), &max_alloc, NULL);
  if (cl_err != CL_SUCCESS ||
      (gdouble) src_rect.width * src_rect.height *
      babl_format_get_bytes_per_pixel (format) > MIN (max_alloc, 1 << 30))
    {
      g_free (areas);
      return FALSE;
    }

  srect.s[0]     = src_rect.x;
  srect.s[1]     = src_rect.y;
  srect.s[2]     = src_rect.width;
  srect.s[3]     = src_rect.height;
  abyss4.s[0]    = abyss->x;
  abyss4.s[1]    = abyss->y;
  abyss4.s[2]    = abyss->width;
  abyss4.s[3]    = abyss->height;
  boundary4.s[0] = boundary->x;
  boundary4.s[1] = boundary->y;
  boundary4.s[2] = boundary->width;
  boundary4.s[3] = boundary->height;
  for (j = 0; j < 4; j++)
    bg.s[j] = background[j];

  areas_tex = gegl_clCreateBuffer (gegl_cl_get_context (),
                                   CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY,
                                   chunks_x * chunks_y * sizeof (cl_int4),
                                   areas, &cl_err);
  g_free (areas);
  if (cl_err != CL_SUCCESS)
    return FALSE;

  in_tex = cl_upload_rect (input, &src_rect, format);
  if (!in_tex)
    {
      gegl_clReleaseMemObject (areas_tex);
      return FALSE;
    }

  i = gegl_buffer_cl_iterator_new (output, result, out_format,
                                   GEGL_CL_BUFFER_WRITE);

  while (gegl_buffer_cl_iterator_next (i, &err))
    {
      size_t global_ws[2];

      if (err)
        goto error_iter;

      global_ws[0] = i->roi[0].width;
      global_ws[1] = i->roi[0].height;

      cl_err = gegl_cl_set_kernel_args (cl_data->kernel[0],
                                        sizeof (cl_mem),    &in_tex,
                                        sizeof (cl_int4),   &srect,
                                        sizeof (cl_int4),   &abyss4,
                                        sizeof (cl_mem),    &i->tex[0],
                                        sizeof (cl_int),    &i->roi[0].x,
                                        sizeof (cl_int),    &i->roi[0].y,
                                        sizeof (cl_int),    &result->x,
                                        sizeof (cl_int),    &result->y,
                                        sizeof (cl_mem),    &areas_tex,
                                        sizeof (cl_int),    &chunks_x,
                                        sizeof (cl_int4),   &boundary4,
                                        sizeof (cl_float4), &bg,
                                        sizeof (cl_float2), &coord_args[0],
                                        sizeof (cl_float2), &coord_args[1],
                                        sizeof (cl_float2), &coord_args[2],
                                        sizeof (cl_float2), &coord_args[3],
                                        sizeof (cl_float2), &coord_args[4],
                                        sizeof (cl_float2), &coord_args[5],
                                        sizeof (cl_float),  &brighten,
                                        sizeof (cl_int),    &unpremultiply,
                                        NULL);
      CL_CHECK;

      cl_err = gegl_clEnqueueNDRangeKernel (gegl_cl_get_command_queue (),
                                            cl_data->kernel[0], 2,
                                            NULL, global_ws, NULL,
                                            0, NULL, NULL);
      CL_CHECK;
    }

  gegl_clReleaseMemObject (in_tex);
  gegl_clReleaseMemObject (areas_tex);

  return TRUE;

error:
  gegl_buffer_cl_iterator_stop (i);
error_iter:
  gegl_clReleaseMemObject (in_tex);
  gegl_clReleaseMemObject (areas_tex);

  return FALSE;
}

static gboolean
process (GeglOperation       *operation,
         GeglBuffer          *input,
         GeglBuffer          *output,
         const GeglRectangle *result,
         gint                 level)
{
  GeglProperties *o      = GEGL_PROPERTIES (operation);
  const Babl     *space  = gegl_operation_get_format (operation, "output");
  const Babl     *format = babl_format_with_space ("RaGaBaA float", space);
  GeglSampler    *sampler;
  LensValues      lens;
  GeglRectangle   boundary;
  gint            i, j;
  gfloat         *src_buf, *dst_buf;
  gfloat          background[4];

  sampler  = gegl_buffer_sampler_new_at_level (input,
                                               babl_format_with_space ("RGBA float", space),
                                               GEGL_SAMPLER_LINEAR, level);
  boundary = *gegl_operation_source_get_bounding_box (operation, "input");
  lens     =  lens_setup_calc (o, boundary);

  gegl_color_get_pixel (o->background, format, background);

  if (level == 0 && gegl_operation_use_opencl (operation) &&
      cl_process (operation, input, output, result, &boundary, &lens,
                  background))
    {
      g_object_unref (sampler);
      return TRUE;
    }

  src_buf = g_new0 (gfloat, SQR (MAX_WH) * 4);
  dst_buf = g_new0 (gfloat, SQR (CHUNK_SIZE) * 4);

  for (j = 0; (j-1) * CHUNK_SIZE < result->height; j++)
    for (i = 0; (i-1) * CHUNK_SIZE < result->width; i++)
      {
        GeglRectangle chunked_result;
        GeglRectangle area;
        gint          x, y;

        chunked_result = *GEGL_RECTANGLE (result->x + i * CHUNK_SIZE,
                                          result->y + j * CHUNK_SIZE,
                                          CHUNK_SIZE, CHUNK_SIZE);

        gegl_rectangle_intersect (&chunked_result, &chunked_result, result);

        if (chunked_result.width < 1  || chunked_result.height < 1)
          continue;

        area = get_required (&boundary, &chunked_result, operation);

        clamp_area (&area, lens.centre_x, lens.centre_y);

        gegl_buffer_get (input, &area, 1.0, format, src_buf,
                         GEGL_AUTO_ROWSTRIDE, GEGL_ABYSS_CLAMP);

        for (y = chunked_result.y; y < chunked_result.y + chunked_result.height; y++)
          for (x = chunked_result.x; x < chunked_result.x + chunked_result.width; x++)
            {
              lens_distort_func (src_buf, dst_buf, &area, &chunked_result, &boundary,
                                 &lens, x, y, sampler, background, level);
            }

        gegl_buffer_set (output, &chunked_result, 0, format,
                         dst_buf, GEGL_AUTO_ROWSTRIDE);
      }

  g_free (dst_buf);
  g_free (src_buf);

  g_object_unref (sampler);

  return TRUE;
}

static void
gegl_op_class_init (GeglOpClass *klass)
{
  GeglOperationClass       *operation_class;
  GeglOperationFilterClass *filter_class;
  gchar                    *composition =
    "<?xml version='1.0' encoding='UTF-8'?>"
    "<gegl>"
    "<node operation='gegl:lens-distortion'>"
    "  <params>"
    "    <param name='main'>100</param>"
    "    <param name='zoom'>20</param>"
    "    <param name='edge'>100</param>"
    "    <param name='x-shift'>20</param>"
    "    <param name='y-shift'>20</param>"
    "  </params>"
    "</node>"
    "<node operation='gegl:load'>"
    "  <params>"
    "    <param name='path'>standard-input.png</param>"
    "  </params>"
    "</node>"
    "</gegl>";

  operation_class = GEGL_OPERATION_CLASS (klass);
  filter_class    = GEGL_OPERATION_FILTER_CLASS (klass);

  operation_class->get_required_for_output = get_required_for_output;

  filter_class->process                    = process;
  operation_class->opencl_support          = TRUE;

  gegl_operation_class_set_keys (operation_class,
    "name",                  "gegl:lens-distortion",
    "title",                 _("Lens Distortion"),
    "categories",            "distort",
    "position-dependent",    "true",
    "license",               "GPL3+",
    "reference-hash" ,       "e2a6800b59b9b29bc1ebe994c3f79928",
    "reference-composition", composition,
    "description", _("Corrects barrel or pincushion lens distortion."),
    NULL);
}

#endif
