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
 * Copyright 2018 Simon Budig <simon@gimp.org>
 * Copyright 2011 Michael Muré <batolettre@gmail.com>
 * Copyright 2011 Robert Sasu <sasu.robert@gmail.com>
 * Copyright 2011 Hans Lo <hansshulo@gmail.com>
 * Copyright 1997 Brian Degenhardt <bdegenha@ucsd.edu>
 */

#include "config.h"
#include <glib/gi18n-lib.h>

#ifdef GEGL_PROPERTIES

enum_start (gegl_ripple_wave_type)
  enum_value (GEGL_RIPPLE_WAVE_TYPE_SINE,     "sine",     N_("Sine"))
  enum_value (GEGL_RIPPLE_WAVE_TYPE_TRIANGLE, "triangle", N_("Triangle"))
  enum_value (GEGL_RIPPLE_WAVE_TYPE_SAWTOOTH, "sawtooth", N_("Sawtooth"))
enum_end (GeglRippleWaveType)

property_double (amplitude, _("Amplitude"), 25.0)
    value_range (0.0, 1000.0)
    ui_gamma    (2.0)

property_double (period, _("Period"), 200.0)
    value_range (0.0, 1000.0)
    ui_gamma    (1.5)

property_double (phi, _("Phase shift"), 0.0)
    value_range (-1.0, 1.0)

property_double (angle, _("Angle"), 0.0)
    value_range (-180, 180)
    ui_meta     ("unit", "degree")
    ui_meta     ("direction", "ccw")

property_enum  (sampler_type, _("Resampling method"),
    GeglSamplerType, gegl_sampler_type, GEGL_SAMPLER_CUBIC)

property_enum (wave_type, _("Wave type"),
    GeglRippleWaveType, gegl_ripple_wave_type, GEGL_RIPPLE_WAVE_TYPE_SINE)

property_enum (abyss_policy, _("Abyss policy"),
               GeglAbyssPolicy, gegl_abyss_policy,
               GEGL_ABYSS_NONE)
    description (_("How image edges are handled"))

property_boolean (tileable, _("Tileable"), FALSE)
    description(_("Retain tilebility"))

#else

#define GEGL_OP_AREA_FILTER
#define GEGL_OP_NAME     ripple
#define GEGL_OP_C_SOURCE ripple.c

#include "gegl-op.h"
#include <stdio.h>
#include <stdlib.h>

static void
prepare (GeglOperation *operation)
{
  GeglProperties              *o;
  GeglOperationAreaFilter *op_area;
  const Babl              *space = gegl_operation_get_source_space (operation, "input");

  op_area = GEGL_OPERATION_AREA_FILTER (operation);
  o       = GEGL_PROPERTIES (operation);

  op_area->left   = o->amplitude;
  op_area->right  = o->amplitude;
  op_area->top    = o->amplitude;
  op_area->bottom = o->amplitude;

  gegl_operation_set_format (operation, "input",
                             babl_format_with_space ("RGBA float", space));
  gegl_operation_set_format (operation, "output",
                             babl_format_with_space ("RGBA float", space));
}

/* The phase of the wave at (x, y), before the phase offset of the wave
 * type and the remainder modulo the period.  The OpenCL path uses the
 * same functions where it needs the exact CPU result, keep them shared.
 */
static inline gdouble
ripple_nx (gdouble angle_rad,
           gint    x,
           gint    y)
{
  return x * cos (angle_rad) - y * sin (angle_rad);
}

/* lambda of the sawtooth wave, in [0, period] */
static inline gdouble
ripple_sawtooth_lambda (gdouble nx,
                        gdouble period,
                        gdouble phi)
{
  gdouble lambda = remainder (nx + period / 2 - phi * period, period);

  if (lambda < 0)
    lambda += period;

  return lambda;
}

static GeglAbyssPolicy
get_abyss_policy (GeglOperation *operation,
                  const gchar   *input_pad)
{
  GeglProperties *o = GEGL_PROPERTIES (operation);

  return o->abyss_policy;
}

#include "opencl/gegl-cl.h"
#include "gegl-buffer-cl-iterator.h"

#include "opencl/ripple.cl.h"

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

  return (gdouble) src_rect->width * src_rect->height <= (1 << 26);
}

/* Splits @v into a hi/lo pair of floats. */
static void
cl_split (gdouble    v,
          cl_float2 *out)
{
  out->s[0] = v;
  out->s[1] = v - out->s[0];
}

/* sin (2 pi v) and cos (2 pi v) for v in [-1/2, 1/2], exact where they
 * are 0 or +-1 */
static void
cl_sincos_cycles (gdouble  v,
                  gdouble *s,
                  gdouble *c)
{
  gdouble n  = rint (4.0 * v);
  gdouble r  = v - n / 4.0; /* exact */
  gdouble sr = sin (2.0 * G_PI * r);
  gdouble cr = cos (2.0 * G_PI * r);

  switch ((gint) n & 3)
    {
    case 0:  *s =  sr; *c =  cr; break;
    case 1:  *s =  cr; *c = -sr; break;
    case 2:  *s = -sr; *c = -cr; break;
    default: *s = -cr; *c =  sr; break;
    }
}

/* Computes the phase table of the kernel: for i in [0, n), the phase of
 * (origin + i) * step + offset reduced modulo @period.  If
 * @cycles, it is converted to units of the period, plus @cycles_offset,
 * and the table holds its sine and cosine as hi/lo pairs; else it holds
 * the phase, in pixels, in [-period / 2, period / 2], as a hi/lo pair.
 */
static cl_mem
cl_phase_table (gint      origin,
                gint      n,
                gdouble   step,
                gdouble   offset,
                gdouble   period,
                gboolean  cycles,
                gdouble   cycles_offset)
{
  cl_int     cl_err = 0;
  cl_mem     tex;
  cl_float4 *table  = g_new0 (cl_float4, n);
  gint       i;

  for (i = 0; i < n; i++)
    {
      gdouble   v = remainder ((origin + i) * step + offset, period);
      cl_float2 hi_lo;

      if (cycles)
        {
          gdouble s, c;

          cl_sincos_cycles (remainder (v / period + cycles_offset, 1.0),
                            &s, &c);

          cl_split (s, &hi_lo);
          table[i].s[0] = hi_lo.s[0];
          table[i].s[1] = hi_lo.s[1];
          cl_split (c, &hi_lo);
          table[i].s[2] = hi_lo.s[0];
          table[i].s[3] = hi_lo.s[1];
        }
      else
        {
          cl_split (v, &hi_lo);
          table[i].s[0] = hi_lo.s[0];
          table[i].s[1] = hi_lo.s[1];
        }
    }

  tex = gegl_clCreateBuffer (gegl_cl_get_context (),
                             CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY,
                             n * sizeof (cl_float4), table, &cl_err);
  g_free (table);

  if (cl_err != CL_SUCCESS)
    return NULL;

  return tex;
}

static gboolean
cl_process (GeglOperation       *operation,
            GeglBuffer          *input,
            GeglBuffer          *output,
            const GeglRectangle *roi,
            gdouble              amplitude,
            gdouble              period,
            gdouble              angle_rad)
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
  cl_int                policy      = o->abyss_policy;
  cl_int                unpremultiply;
  cl_int                wave_type   = o->wave_type;
  cl_mem                phase_x     = NULL;
  cl_mem                phase_y     = NULL;
  cl_int                phase_x0    = roi->x;
  cl_int                phase_y0    = roi->y;
  gdouble               cos_a       = cos (angle_rad);
  gdouble               sin_a       = sin (angle_rad);
  gdouble               offset;
  gboolean              cycles;
  cl_float2             f_amplitude;
  cl_float2             f_period;
  cl_float2             cos_angle;
  cl_float2             sin_angle;

  if (format != babl_format_with_space ("RGBA float", space))
    return FALSE;

  /* as hi/lo pairs, for the kernel to compute the coordinates precisely */
  cl_split (amplitude, &f_amplitude);
  cl_split (period,    &f_period);
  cl_split (cos_a,     &cos_angle);
  cl_split (sin_a,     &sin_angle);

  /* the phase offset that process () adds to nx, before the remainder */
  switch (o->wave_type)
    {
    case GEGL_RIPPLE_WAVE_TYPE_SAWTOOTH:
      offset = period / 2 - o->phi * period;
      cycles = FALSE;
      break;
    case GEGL_RIPPLE_WAVE_TYPE_TRIANGLE:
      offset = period * 3 / 4 - o->phi * period;
      cycles = FALSE;
      break;
    default:
      offset = 0.0;
      cycles = TRUE;
      break;
    }

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

  /* samples are at most amplitude pixels away, and the samplers read up
   * to two pixels beyond the sampled one */
  margin = ceil (fabs (amplitude)) + 3;
  needed = *roi;
  needed.x      -= margin;
  needed.y      -= margin;
  needed.width  += 2 * margin;
  needed.height += 2 * margin;

  if (! cl_get_source_rect (&needed, abyss, policy, &src_rect))
    return FALSE;

  if (!cl_data)
    {
      const char *kernel_name[] = {"ripple_nearest", "ripple_linear",
                                   "ripple_cubic", "ripple_nohalo", NULL};
      cl_data = gegl_cl_compile_and_build (ripple_cl_source, kernel_name);
    }

  if (!cl_data)
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

  /* nx = x * cos (angle) - y * sin (angle), split into a column and a row
   * term, each reduced on the host in double precision */
  phase_x = cl_phase_table (roi->x, roi->width, cos_a, offset, period,
                            cycles, o->phi);
  phase_y = cl_phase_table (roi->y, roi->height, -sin_a, 0.0, period,
                            cycles, 0.0);
  if (!phase_x || !phase_y)
    goto error_iter;

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
                                        sizeof (cl_int),   &unpremultiply,
                                        sizeof (cl_int),   &wave_type,
                                        sizeof (cl_mem),   &phase_x,
                                        sizeof (cl_mem),   &phase_y,
                                        sizeof (cl_int),   &phase_x0,
                                        sizeof (cl_int),   &phase_y0,
                                        sizeof (cl_float2), &f_amplitude,
                                        sizeof (cl_float2), &f_period,
                                        sizeof (cl_float2), &cos_angle,
                                        sizeof (cl_float2), &sin_angle,
                                        NULL);
      CL_CHECK;

      cl_err = gegl_clEnqueueNDRangeKernel (gegl_cl_get_command_queue (),
                                            kernel, 2,
                                            NULL, global_ws, NULL,
                                            0, NULL, NULL);
      CL_CHECK;
    }

  gegl_clReleaseMemObject (in_tex);
  gegl_clReleaseMemObject (phase_x);
  gegl_clReleaseMemObject (phase_y);

  return TRUE;

error:
  gegl_buffer_cl_iterator_stop (i);
error_iter:
  gegl_clReleaseMemObject (in_tex);
  if (phase_x)
    gegl_clReleaseMemObject (phase_x);
  if (phase_y)
    gegl_clReleaseMemObject (phase_y);

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
  GeglSampler        *sampler = gegl_buffer_sampler_new_at_level (input,
                                                         format,
                                                         o->sampler_type,
                                                         level);
  GeglBufferIterator *iter;
  gdouble angle_rad, period, amplitude, phi;

  angle_rad = o->angle / 180.0 * G_PI;
  period    = o->period;
  amplitude = o->amplitude;
  phi       = o->phi;

  if (period < 0.0001)
    {
      period = 1.0;
      amplitude = 0.0;
    }

  if (o->tileable)
    {
      gint w, h;
      gdouble n, m;
      GeglRectangle *bbox;

      bbox = gegl_operation_source_get_bounding_box (operation, "input");
      w = bbox->width;
      h = bbox->height;

      n = cos (angle_rad) * w / period;
      m = sin (angle_rad) * h / period;

      /* round away from zero */
      n = round (n);
      m = round (m);
      if (n == 0 && m == 0)
        {
          n = 1;
          amplitude = 0;
        }

      /* magic! */
      angle_rad = atan2 (m * w, h * n);
      period = sqrt ((gdouble) h*h * w*w / (n*n * h*h + m*m * w*w));

      /* ok, not actually.
       *
       * For the result of the ripple op being tileable you need
       * to have the period/angle select in a way, so that the top left
       * corner has an integer * period distance along the angle to
       * the top right corner as well as the bottom left corner.
       *
       * I.e.:
       *
       * cos (angle) * width = n * period
       * sin (angle) * height = m * period
       *
       * with n, m being integers.
       *
       * We determine n, m by rounding the results optained by the
       * user specified angle/period and then calculate a hopefully only
       * slightly modified new angle/period that meets these criteria.
       *
       * We determine the angle by computing tan(), thereby eliminating
       * the period, then determining the period via a formula
       * derived from the  sin²(a)+cos²(a) = 1  identity.
       */
    }

  if (level == 0 && gegl_operation_use_opencl (operation) &&
      cl_process (operation, input, output, result,
                  amplitude, period, angle_rad))
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
            gdouble shift;
            gdouble coordsx;
            gdouble coordsy;
            gdouble lambda;

            gdouble nx = ripple_nx (angle_rad, x, y);

            switch (o->wave_type)
              {
                case GEGL_RIPPLE_WAVE_TYPE_SAWTOOTH:
                  lambda = ripple_sawtooth_lambda (nx, period, phi);
                  shift = amplitude * (((lambda / period) * 2) - 1);
                  break;

                case GEGL_RIPPLE_WAVE_TYPE_TRIANGLE:
                  lambda = remainder (nx + period * 3 / 4 - phi * period, period);
                  if (lambda < 0)
                    lambda += period;
                  shift = amplitude * (fabs (((lambda / period) * 4) - 2) - 1);
                  break;

                case GEGL_RIPPLE_WAVE_TYPE_SINE:
                default:
                  shift = amplitude * sin (2.0 * G_PI * nx / period + 2.0 * G_PI * phi);
                  break;
              }

            coordsx = x + shift * sin (angle_rad);
            coordsy = y + shift * cos (angle_rad);

            gegl_sampler_get (sampler,
                              coordsx,
                              coordsy,
                              NULL,
                              out_pixel,
                              o->abyss_policy);

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
    "name",               "gegl:ripple",
    "title",              _("Ripple"),
    "categories",         "distort",
    "position-dependent", "true",
    "license",            "GPL3+",
    "reference-hash",     "1079777983713a28a14d464a0d391a37",
    "description", _("Displace pixels in a ripple pattern"),
    NULL);
}

#endif
