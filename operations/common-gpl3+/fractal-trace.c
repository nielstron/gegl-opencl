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
 * Copyright (C) 1997 Hirotsuna Mizuno <s1041150@u-aizu.ac.jp>
 * Copyright (C) 2011 Robert Sasu <sasu.robert@gmail.com>
 */

#include "config.h"
#include <glib/gi18n-lib.h>

#ifdef GEGL_PROPERTIES

enum_start (gegl_fractal_trace_type)
  enum_value (GEGL_FRACTAL_TRACE_TYPE_MANDELBROT, "mandelbrot", N_("Mandelbrot"))
  enum_value (GEGL_FRACTAL_TRACE_TYPE_JULIA,      "julia",      N_("Julia"))
enum_end (GeglFractalTraceType)

property_enum (fractal, _("Fractal type"),
               GeglFractalTraceType, gegl_fractal_trace_type,
               GEGL_FRACTAL_TRACE_TYPE_MANDELBROT)

property_double (X1, _("X1"), -1.0)
  description   (_("X1 value, position"))
  value_range   (-50.0, 50.0)
  ui_range      (-2.0, 2.0)

property_double (X2, _("X2"), 0.50)
  description   (_("X2 value, position"))
  value_range   (-50.0, 50.0)
  ui_range      (-2.0, 2.0)

property_double (Y1, _("Y1"), -1.0)
  description   (_("Y1 value, position"))
  value_range   (-50.0, 50.0)
  ui_range      (-2.0, 2.0)

property_double (Y2, _("Y2"), 1.0)
  description   (_("Y2 value, position"))
  value_range   (-50.0, 50.0)
  ui_range      (-2.0, 2.0)

property_double (JX, _("JX"), 0.5)
  description (_("Julia seed X value, position"))
  value_range   (-50.0, 50.0)
  ui_range      (-2.0, 2.0)
  ui_meta       ("visible", "fractal {julia}")

property_double (JY, _("JY"), 0.5)
  description (_("Julia seed Y value, position"))
  value_range   (-50.0, 50.0)
  ui_range      (-2.0, 2.0)
  ui_meta       ("visible", "$JX.visible")

property_int    (depth, _("Depth"), 3)
  value_range   (1, 65536)
  ui_range      (1, 50)

property_double (bailout, _("Bailout length"), 10000.0)
  value_range   (0.0, G_MAXDOUBLE)
  ui_range      (0.0, 10000.0)

property_enum   (abyss_policy, _("Abyss policy"),
                   GeglAbyssPolicy, gegl_abyss_policy, GEGL_ABYSS_LOOP)
  description   (_("How to deal with pixels outside of the input buffer"))

#else

#define GEGL_OP_FILTER
#define GEGL_OP_NAME     fractal_trace
#define GEGL_OP_C_SOURCE fractal-trace.c

#include "gegl-op.h"
#include "gegl-parallel.h"
#include "opencl/gegl-cl.h"
#include "opencl/gegl-buffer-cl-iterator.h"
#include "opencl/fractal-trace.cl.h"

typedef struct
{
  const GeglProperties *properties;
  GeglRectangle         picture;
  gdouble               scale_x, scale_y;
  gdouble               bailout2;
} Fractal;

static void
julia (gdouble  x,
       gdouble  y,
       gdouble  jx,
       gdouble  jy,
       gdouble *u,
       gdouble *v,
       gint     depth,
       gdouble  bailout2)
{
  gint    i;
  gdouble xx = x;
  gdouble yy = y;

  for (i = 0; i < depth; i++)
    {
      gdouble x2, y2, tmp;

      x2 = xx * xx;
      y2 = yy * yy;
      tmp = x2 - y2 + jx;
      yy  = 2 * xx * yy + jy;
      xx  = tmp;

      if ((x2 + y2) > bailout2)
        break;
    }

  *u = xx;
  *v = yy;
}

static inline void
unmap (const Fractal *fractal,
       gdouble        x,
       gdouble        y,
       gdouble       *px,
       gdouble       *py)
{
  const GeglProperties *o = fractal->properties;
  const gdouble cx = o->X1 + (x - fractal->picture.x) * fractal->scale_x;
  const gdouble cy = o->Y1 + (y - fractal->picture.y) * fractal->scale_y;
  const gboolean mandelbrot = o->fractal == GEGL_FRACTAL_TRACE_TYPE_MANDELBROT;
  gdouble rx, ry;

  julia (cx, cy, mandelbrot ? cx : o->JX, mandelbrot ? cy : o->JY,
         &rx, &ry, o->depth, fractal->bailout2);
  *px = (rx - o->X1) / fractal->scale_x + fractal->picture.x;
  *py = (ry - o->Y1) / fractal->scale_y + fractal->picture.y;
}

static inline void
map_pixel (const Fractal     *fractal,
           gint               x,
           gint               y,
           gdouble           *px,
           gdouble           *py,
           GeglBufferMatrix2 *scale)
{
#define gegl_unmap(u,v,ud,vd)                     \
  {                                               \
    gdouble sx, sy;                               \
    unmap (fractal, u, v, &sx, &sy);              \
    (ud) = sx;                                    \
    (vd) = sy;                                    \
  }
  gegl_sampler_compute_scale ((*scale), x, y);
#undef gegl_unmap
  unmap (fractal, x, y, px, py);
}

typedef struct
{
  cl_int2   anchor;
  cl_float2 fraction;
  cl_float2 step_x;
  cl_float2 step_y;
  cl_int2   samples;
} SamplingGeometry;

G_STATIC_ASSERT (sizeof (SamplingGeometry) == 5 * sizeof (cl_int2));

typedef struct
{
  const Fractal        *fractal;
  const GeglRectangle  *abyss;
  GeglRectangle         rect;
  SamplingGeometry     *geometry;
  gint                  unsupported;
} GeometryRows;

static void
geometry_rows (gsize    offset,
               gsize    count,
               gpointer data)
{
  GeometryRows *rows = data;
  SamplingGeometry *geometry = rows->geometry + offset * rows->rect.width;
  const gint first_y = rows->rect.y + (gint) offset;
  const gint last_y = first_y + (gint) count;

  for (gint y = first_y; y < last_y; y++)
    for (gint col = 0; col < rows->rect.width; col++, geometry++)
      {
        GeglBufferMatrix2 scale;
        gdouble x, y_pos;
        gdouble ux = 0, uy = 0, vx = 0, vy = 0;
        gint nx = 1, ny = 1;
        gdouble u2, v2;
        gint ax, ay;

        map_pixel (rows->fractal, rows->rect.x + col, y, &x, &y_pos, &scale);
        x = isfinite (x) ? x : 0;
        y_pos = isfinite (y_pos) ? y_pos : 0;

        u2 = scale.coeff[0][0] * scale.coeff[0][0] +
             scale.coeff[1][0] * scale.coeff[1][0];
        v2 = scale.coeff[0][1] * scale.coeff[0][1] +
             scale.coeff[1][1] * scale.coeff[1][1];
        if (!isfinite (u2) || !isfinite (v2))
          goto unsupported;

        if (u2 >= 4.0 || v2 >= 4.0)
          {
            nx = CLAMP (sqrt (u2), 1.0, 5.0);
            ny = CLAMP (sqrt (v2), 1.0, 5.0);
            ux = scale.coeff[0][0] * (1.0 / nx);
            uy = scale.coeff[1][0] * (1.0 / nx);
            vx = scale.coeff[0][1] * (1.0 / ny);
            vy = scale.coeff[1][1] * (1.0 / ny);
            x -= (scale.coeff[0][0] - ux + scale.coeff[0][1] - vx) / 2.0;
            y_pos -= (scale.coeff[1][0] - uy + scale.coeff[1][1] - vy) / 2.0;
          }

        /* Keep the integral position separate so far-away samples do not
         * lose their fractional offset on devices with no double support.
         * At very large positions the CPU's float-to-int anchor can round
         * across a pixel boundary, so use the CPU for those warps. */
        x -= 0.5;
        y_pos -= 0.5;
        if (fabs (x) + 4.0 * (fabs (ux) + fabs (vx)) >= 0x1p17 ||
            fabs (y_pos) + 4.0 * (fabs (uy) + fabs (vy)) >= 0x1p17 ||
            fabs (ux) + fabs (vx) > 512.0 || fabs (uy) + fabs (vy) > 512.0)
          goto unsupported;

        ax = (gint) floorf ((gfloat) x);
        ay = (gint) floorf ((gfloat) y_pos);
        geometry->anchor.s[0] = ax - rows->abyss->x;
        geometry->anchor.s[1] = ay - rows->abyss->y;
        geometry->fraction.s[0] = x - ax;
        geometry->fraction.s[1] = y_pos - ay;
        geometry->step_x.s[0] = ux;
        geometry->step_x.s[1] = uy;
        geometry->step_y.s[0] = vx;
        geometry->step_y.s[1] = vy;
        geometry->samples.s[0] = nx;
        geometry->samples.s[1] = ny;
      }

  return;

unsupported:
  g_atomic_int_set (&rows->unsupported, TRUE);
}

#define GEOMETRY_PIXELS (128 * 1024)

G_LOCK_DEFINE_STATIC (fractal_cl);

static gboolean
cl_process (const Fractal       *fractal,
            GeglBuffer          *input,
            GeglBuffer          *output,
            const GeglRectangle *result)
{
  static GeglClRunData *cl_data = NULL;
  const GeglRectangle *abyss = gegl_buffer_get_abyss (input);
  const Babl *premultiplied = babl_format_with_space ("RaGaBaA float",
                                                     gegl_buffer_get_format (input));
  const cl_command_queue queue = gegl_cl_get_command_queue ();
  const cl_context context = gegl_cl_get_context ();
  const cl_int2 size = {{abyss->width, abyss->height}};
  const cl_int policy = fractal->properties->abyss_policy;
  const gfloat abyss_rgba[4] = {policy == GEGL_ABYSS_WHITE,
                               policy == GEGL_ABYSS_WHITE,
                               policy == GEGL_ABYSS_WHITE,
                               policy == GEGL_ABYSS_WHITE || policy == GEGL_ABYSS_BLACK};
  cl_float4 abyss_color;
  size_t src_bytes;
  cl_int cl_err;
  cl_mem src = NULL, geo = NULL;
  gpointer mapped;
  SamplingGeometry *geometry = NULL;
  GeglBufferClIterator *iter = NULL;
  gboolean err = FALSE;
  gboolean success = FALSE;

  if (abyss->width <= 0 || abyss->height <= 0 ||
      ABS ((gint64) abyss->x) >= 0x400000 || ABS ((gint64) abyss->y) >= 0x400000 ||
      (gsize) abyss->width > G_MAXINT / sizeof (cl_float4) ||
      (gsize) abyss->width > G_MAXSIZE / sizeof (cl_float4) / abyss->height)
    return FALSE;
  src_bytes = (gsize) abyss->width * abyss->height * sizeof (cl_float4);

  G_LOCK (fractal_cl);
  if (!cl_data)
    {
      const char *kernels[] = {"fractal_trace", NULL};
      cl_data = gegl_cl_compile_and_build (fractal_trace_cl_source, kernels);
    }
  G_UNLOCK (fractal_cl);
  if (!cl_data)
    return FALSE;

  babl_process (babl_fish (babl_format ("RGBA float"), premultiplied),
                abyss_rgba, &abyss_color, 1);
  src = gegl_clCreateBuffer (context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY,
                             src_bytes, NULL, &cl_err);
  CL_CHECK;
  mapped = gegl_clEnqueueMapBuffer (queue, src, CL_TRUE, CL_MAP_WRITE,
                                    0, src_bytes, 0, NULL, NULL, &cl_err);
  CL_CHECK;
  gegl_buffer_get (input, abyss, 1.0, premultiplied, mapped,
                   GEGL_AUTO_ROWSTRIDE, GEGL_ABYSS_NONE);
  cl_err = gegl_clEnqueueUnmapMemObject (queue, src, mapped, 0, NULL, NULL);
  CL_CHECK;

  geo = gegl_clCreateBuffer (context, CL_MEM_READ_ONLY,
                             GEOMETRY_PIXELS * sizeof (SamplingGeometry),
                             NULL, &cl_err);
  CL_CHECK;
  geometry = g_new (SamplingGeometry, GEOMETRY_PIXELS);

  iter = gegl_buffer_cl_iterator_new (output, result, premultiplied, GEGL_CL_BUFFER_WRITE);
  while (gegl_buffer_cl_iterator_next (iter, &err))
    {
      const GeglRectangle chunk = iter->roi[0];
      const gint rows_per_band = GEOMETRY_PIXELS / MAX (chunk.width, 1);

      if (rows_per_band < 1)
        goto error;

      for (gint row = 0; row < chunk.height; row += rows_per_band)
        {
          const gint height = MIN (rows_per_band, chunk.height - row);
          const cl_int start = row * chunk.width;
          const size_t work = (size_t) chunk.width * height;
          GeometryRows rows = {fractal, abyss, {chunk.x, chunk.y + row,
                                                chunk.width, height}, geometry, FALSE};

          gegl_parallel_distribute_range (height, 2048.0 / chunk.width,
                                          geometry_rows, &rows);
          if (g_atomic_int_get (&rows.unsupported))
            goto error;

          cl_err = gegl_clEnqueueWriteBuffer (queue, geo, CL_TRUE, 0,
                                              work * sizeof (SamplingGeometry),
                                              geometry, 0, NULL, NULL);
          CL_CHECK;
          G_LOCK (fractal_cl);
          cl_err = gegl_cl_set_kernel_args (cl_data->kernel[0],
                                            sizeof (src), &src,
                                            sizeof (size), &size,
                                            sizeof (policy), &policy,
                                            sizeof (abyss_color), &abyss_color,
                                            sizeof (geo), &geo,
                                            sizeof (cl_mem), &iter->tex[0],
                                            sizeof (start), &start, NULL);
          if (cl_err == CL_SUCCESS)
            cl_err = gegl_clEnqueueNDRangeKernel (queue, cl_data->kernel[0], 1,
                                                  NULL, &work, NULL, 0, NULL, NULL);
          G_UNLOCK (fractal_cl);
          CL_CHECK;
        }
    }
  iter = NULL;
  success = !err;

error:
  if (iter)
    gegl_buffer_cl_iterator_stop (iter);
  if (geo)
    gegl_clReleaseMemObject (geo);
  if (src)
    gegl_clReleaseMemObject (src);
  g_free (geometry);
  return success;
}

static gboolean
process (GeglOperation       *operation,
         GeglBuffer          *input,
         GeglBuffer          *output,
         const GeglRectangle *result,
         gint                 level)
{
  GeglProperties    *o = GEGL_PROPERTIES (operation);
  const GeglRectangle boundary = gegl_operation_get_bounding_box (operation);
  const Fractal fractal = {o, boundary,
                           (o->X2 - o->X1) / boundary.width,
                           (o->Y2 - o->Y1) / boundary.height,
                           o->bailout * o->bailout};
  const Babl    *format = babl_format_with_space ("RGBA float",
                              gegl_operation_get_format (operation, "output"));
  GeglSampler   *sampler;
  gfloat        *dst_buf;

  if (level == 0 && gegl_operation_use_opencl (operation) &&
      cl_process (&fractal, input, output, result))
    return TRUE;

  dst_buf = g_new (gfloat, (gsize) result->width * result->height * 4);
  sampler = gegl_buffer_sampler_new_at_level (input, format, GEGL_SAMPLER_CUBIC, level);

  for (gint y = 0; y < result->height; y++)
    for (gint x = 0; x < result->width; x++)
      {
        gdouble px, py;
        GeglBufferMatrix2 scale;
        map_pixel (&fractal, result->x + x, result->y + y, &px, &py, &scale);
        gegl_sampler_get (sampler, px, py, &scale,
                          dst_buf + ((gsize) y * result->width + x) * 4,
                          o->abyss_policy);
      }

  gegl_buffer_set (output, result, 0, format, dst_buf, GEGL_AUTO_ROWSTRIDE);
  g_object_unref (sampler);

  g_free (dst_buf);

  return TRUE;
}

static GeglRectangle
get_required_for_output (GeglOperation       *operation,
                         const gchar         *input_pad,
                         const GeglRectangle *roi)
{
  const GeglRectangle *in_rect =
      gegl_operation_source_get_bounding_box (operation, "input");

  if (in_rect && ! gegl_rectangle_is_infinite_plane (in_rect))
    return *in_rect;

  return *roi;
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
  operation_class->opencl_support          = TRUE;

  filter_class->process                    = process;

  gegl_operation_class_set_keys (operation_class,
    "name",               "gegl:fractal-trace",
    "title",              _("Fractal Trace"),
    "position-dependent", "true",
    "categories",         "map",
    "license",            "GPL3+",
    "reference-hash",     "af898bde4770aae8041422e159f5fc34",
    "description", _("Transform the image with the fractals"),
    NULL);
}

#endif
