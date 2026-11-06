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
 * Copyright (C) 2017 Ell
 */

#include "config.h"
#include <glib/gi18n-lib.h>
#if defined(_WIN32) && !defined(read)
#include <io.h>
#define read _read
#endif

#ifdef GEGL_PROPERTIES

property_double (radius, _("Radius"), 10.0)
  description (_("Blur radius"))
  value_range (0.0, G_MAXDOUBLE)
  ui_range    (0.0, 100.0)
  ui_gamma    (2.0)
  ui_meta     ("unit", "pixel-distance")

property_double (highlight_factor, _("Highlight factor"), 0.0)
  description (_("Relative highlight strength"))
  value_range (0.0, 1.0)

property_double (highlight_threshold_low, _("Highlight threshold (low)"), 0.0)
  ui_range    (0.0, 1.0)
  ui_meta     ("role", "range-start")
  ui_meta     ("unit", "luminance")
  ui_meta     ("range-label", _("Highlight threshold"))

property_double (highlight_threshold_high, _("Highlight threshold (high)"), 1.0)
  ui_range    (0.0, 1.0)
  ui_meta     ("role", "range-end")
  ui_meta     ("unit", "luminance")

property_boolean (clip, _("Clip to input extents"), TRUE)
  description (_("Clip output to the input extents"))

property_boolean (linear_mask, _("Linear mask"), FALSE)
  description (_("Use linear mask values"))

#else

#define GEGL_OP_COMPOSER
#define GEGL_OP_NAME     lens_blur
#define GEGL_OP_C_SOURCE lens-blur.cc

#include "gegl-op.h"

static void
prepare (GeglOperation *operation)
{
  GeglProperties *o = GEGL_PROPERTIES (operation);
  const Babl     *space;
  const Babl     *format;

  space  = gegl_operation_get_source_space (operation, "input");
  format = babl_format_with_space ("RGBA float", space);

  gegl_operation_set_format (operation, "input",  format);
  gegl_operation_set_format (operation, "output", format);

  gegl_operation_set_format (operation, "aux",
                             babl_format_with_space (
                               o->linear_mask ? "Y float" : "Y' float",
                               gegl_operation_get_source_space (operation,
                                                                "aux")));

  o->user_data = (gpointer) babl_fish (format,
                                       babl_format_with_space ("Y float",
                                                               space));
}

static GeglRectangle
get_bounding_box (GeglOperation *operation)
{
  GeglProperties      *o      = GEGL_PROPERTIES (operation);
  const GeglRectangle *in_rect;
  GeglRectangle        result = {};

  in_rect = gegl_operation_source_get_bounding_box (operation, "input");

  if (in_rect)
    {
      result = *in_rect;

      if (! o->clip)
        {
          gint iradius = floor (o->radius + 0.5);

          result.x      -= iradius;
          result.y      -= iradius;
          result.width  += 2 * iradius;
          result.height += 2 * iradius;
        }
    }

  return result;
}

static GeglRectangle
get_required_for_output (GeglOperation       *operation,
                         const gchar         *input_pad,
                         const GeglRectangle *roi)
{
  GeglProperties *o       = GEGL_PROPERTIES (operation);
  GeglRectangle   result  = *roi;
  gint            iradius = floor (o->radius + 0.5);

  result.x      -= iradius;
  result.y      -= iradius;
  result.width  += 2 * iradius;
  result.height += 2 * iradius;

  return result;
}

static gboolean
operation_process (GeglOperation        *operation,
                   GeglOperationContext *context,
                   const gchar          *output_prop,
                   const GeglRectangle  *roi,
                   gint                  level)
{
  GeglProperties *o = GEGL_PROPERTIES (operation);

  if (o->radius < 0.5)
    {
      gegl_operation_context_set_object (
        context, "output",
        gegl_operation_context_get_object (context, "input"));

      return TRUE;
    }

  return GEGL_OPERATION_CLASS (gegl_op_parent_class)->process (
    operation, context, output_prop, roi, level);
}

#include "opencl/gegl-cl.h"
#include "gegl-buffer-cl-iterator.h"

#include "opencl/lens-blur.cl.h"

/* the largest scan (local work) size used by lens_blur_prefix */
#define CL_MAX_SCAN_SIZE 256

static GeglClRunData *cl_data = NULL;

static gboolean
cl_lens_blur (cl_mem               in_tex,
              cl_mem               out_tex,
              cl_mem               spans_tex,
              const GeglRectangle *roi,
              const GeglRectangle *src_rect,
              const cl_int4       *clip,
              const cl_float4     *luminance,
              cl_float             threshold_low,
              cl_float             threshold_high,
              cl_float             factor,
              cl_float             highlight_max,
              cl_int               iradius)
{
  cl_int   cl_err     = 0;
  cl_mem   prefix_tex = NULL;
  cl_mem   prefix_w   = NULL;
  size_t   n          = (size_t) (src_rect->width + 1) * src_rect->height;
  size_t   scan_size;
  size_t   max_size;
  cl_ulong local_mem;
  size_t   global_ws[2];
  size_t   local_ws[2];

  if (!cl_data)
    {
      const char *kernel_name[] = { "lens_blur_prefix", "lens_blur_disc",
                                    NULL };
      cl_data = gegl_cl_compile_and_build (lens_blur_cl_source, kernel_name);
    }

  if (!cl_data)
    return TRUE;

  /* the scan size is a power of two within the kernel's work group size
   * and local memory limits
   */
  cl_err = gegl_clGetKernelWorkGroupInfo (cl_data->kernel[0],
                                          gegl_cl_get_device (),
                                          CL_KERNEL_WORK_GROUP_SIZE,
                                          sizeof (size_t), &max_size,
                                          NULL);
  CL_CHECK;

  cl_err = gegl_clGetDeviceInfo (gegl_cl_get_device (),
                                 CL_DEVICE_LOCAL_MEM_SIZE,
                                 sizeof (cl_ulong), &local_mem,
                                 NULL);
  CL_CHECK;

  for (scan_size = CL_MAX_SCAN_SIZE;
       scan_size > 1 &&
       (scan_size > max_size ||
        scan_size * (2 * sizeof (cl_float4) + sizeof (cl_float2)) >
        local_mem / 2);
       scan_size /= 2);

  prefix_tex = gegl_clCreateBuffer (gegl_cl_get_context (),
                                    CL_MEM_READ_WRITE,
                                    2 * n * sizeof (cl_float4), NULL, &cl_err);
  CL_CHECK;

  prefix_w = gegl_clCreateBuffer (gegl_cl_get_context (),
                                  CL_MEM_READ_WRITE,
                                  n * sizeof (cl_float2), NULL, &cl_err);
  CL_CHECK;

  global_ws[0] = scan_size;
  global_ws[1] = src_rect->height;
  local_ws[0]  = scan_size;
  local_ws[1]  = 1;

  cl_err = gegl_cl_set_kernel_args (cl_data->kernel[0],
                                    sizeof (cl_mem),    &in_tex,
                                    sizeof (cl_mem),    &prefix_tex,
                                    sizeof (cl_mem),    &prefix_w,
                                    2 * scan_size * sizeof (cl_float4), NULL,
                                    scan_size * sizeof (cl_float2),     NULL,
                                    sizeof (cl_int),    &src_rect->width,
                                    sizeof (cl_int4),   clip,
                                    sizeof (cl_float4), luminance,
                                    sizeof (cl_float),  &threshold_low,
                                    sizeof (cl_float),  &threshold_high,
                                    sizeof (cl_float),  &factor,
                                    sizeof (cl_float),  &highlight_max,
                                    NULL);
  CL_CHECK;

  cl_err = gegl_clEnqueueNDRangeKernel (gegl_cl_get_command_queue (),
                                        cl_data->kernel[0], 2,
                                        NULL, global_ws, local_ws,
                                        0, NULL, NULL);
  CL_CHECK;

  global_ws[0] = roi->width;
  global_ws[1] = roi->height;

  cl_err = gegl_cl_set_kernel_args (cl_data->kernel[1],
                                    sizeof (cl_mem), &prefix_tex,
                                    sizeof (cl_mem), &prefix_w,
                                    sizeof (cl_mem), &out_tex,
                                    sizeof (cl_mem), &spans_tex,
                                    sizeof (cl_int), &src_rect->width,
                                    sizeof (cl_int), &iradius,
                                    NULL);
  CL_CHECK;

  cl_err = gegl_clEnqueueNDRangeKernel (gegl_cl_get_command_queue (),
                                        cl_data->kernel[1], 2,
                                        NULL, global_ws, NULL,
                                        0, NULL, NULL);
  CL_CHECK;

  gegl_clReleaseMemObject (prefix_w);
  gegl_clReleaseMemObject (prefix_tex);

  return FALSE;

error:
  if (prefix_w)
    gegl_clReleaseMemObject (prefix_w);
  if (prefix_tex)
    gegl_clReleaseMemObject (prefix_tex);

  return TRUE;
}

static gboolean
cl_process (GeglOperation       *operation,
            GeglBuffer          *input,
            GeglBuffer          *output,
            const GeglRectangle *roi)
{
  GeglProperties      *o       = GEGL_PROPERTIES (operation);
  const Babl          *format  = gegl_operation_get_format (operation, "input");
  const GeglRectangle *bbox    = gegl_operation_source_get_bounding_box (operation,
                                                                         "input");
  gfloat               radius  = o->radius;
  gint                 iradius = floorf (radius + 0.5f);
  gfloat               highlight_factor;
  gdouble              lum[3];
  cl_float4            luminance;
  cl_int              *spans;
  cl_mem               spans_tex;
  cl_int               cl_err  = 0;
  gint                 read;
  gint                 err     = 0;
  gint                 r;
  GeglBufferClIterator *i;

  highlight_factor = 10.0 * o->highlight_factor * G_LN2;

  babl_space_get_rgb_luminance (babl_format_get_space (format),
                                &lum[0], &lum[1], &lum[2]);
  luminance.s[0] = lum[0];
  luminance.s[1] = lum[1];
  luminance.s[2] = lum[2];
  luminance.s[3] = 0.0f;

  /* the half widths of the disc rows, computed like in process () */
  spans = g_new (cl_int, iradius + 1);

  for (r = 0; r <= iradius; r++)
    spans[r] = sqrtf ((radius + 0.5f) * (radius + 0.5f) - r * r);

  spans_tex = gegl_clCreateBuffer (gegl_cl_get_context (),
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   (iradius + 1) * sizeof (cl_int),
                                   spans, &cl_err);
  g_free (spans);

  if (cl_err != CL_SUCCESS)
    return FALSE;

  i = gegl_buffer_cl_iterator_new (output, roi, format, GEGL_CL_BUFFER_WRITE);

  read = gegl_buffer_cl_iterator_add_2 (i, input, roi, format,
                                        GEGL_CL_BUFFER_READ,
                                        iradius, iradius, iradius, iradius,
                                        GEGL_ABYSS_NONE);

  while (gegl_buffer_cl_iterator_next (i, &err) && ! err)
    {
      const GeglRectangle *src_rect = &i->roi[read];
      cl_int4              clip;

      if (o->clip && bbox)
        {
          clip.s[0] = bbox->x                - src_rect->x;
          clip.s[1] = bbox->y                - src_rect->y;
          clip.s[2] = bbox->x + bbox->width  - src_rect->x;
          clip.s[3] = bbox->y + bbox->height - src_rect->y;
        }
      else
        {
          clip.s[0] = 0;
          clip.s[1] = 0;
          clip.s[2] = src_rect->width;
          clip.s[3] = src_rect->height;
        }

      err = cl_lens_blur (i->tex[read], i->tex[0], spans_tex,
                          &i->roi[0], src_rect, &clip, &luminance,
                          o->highlight_threshold_low,
                          o->highlight_threshold_high,
                          highlight_factor,
                          expf (highlight_factor),
                          iradius);

      if (err)
        {
          gegl_buffer_cl_iterator_stop (i);
          break;
        }
    }

  gegl_clReleaseMemObject (spans_tex);

  return ! err;
}

static gboolean
process (GeglOperation       *operation,
         GeglBuffer          *input,
         GeglBuffer          *aux,
         GeglBuffer          *output,
         const GeglRectangle *roi,
         gint                 level)
{
  GeglProperties *o           = GEGL_PROPERTIES (operation);
  const Babl     *format      = gegl_operation_get_format (operation, "input");
  const Babl     *aux_format  = gegl_operation_get_format (operation, "aux");
  const Babl     *weight_fish = (const Babl *) o->user_data;
  GeglRectangle   rect;
  gfloat         *in;
  gfloat         *in_w;
  gfloat         *out;
  gfloat         *out_w;
  gfloat         *mask        = NULL;
  gfloat          highlight_threshold_low;
  gfloat          highlight_threshold_high;
  gfloat          highlight_factor;
  gfloat          highlight_max;
  gfloat          radius      = o->radius;
  gint            iradius     = floorf (radius + 0.5f);
  gint            size        = 2 * iradius + 1;
  gint            y;

  /* the OpenCL implementation does not support masks */
  if (! aux && gegl_operation_use_opencl (operation))
    if (cl_process (operation, input, output, roi))
      return TRUE;

  highlight_threshold_low  = o->highlight_threshold_low;
  highlight_threshold_high = o->highlight_threshold_high;
  highlight_factor         = 10.0 * o->highlight_factor * G_LN2;
  highlight_max            = expf (highlight_factor);

  rect.x      = roi->x      -     iradius;
  rect.y      = roi->y      -     iradius;
  rect.width  = roi->width  + 2 * iradius;
  rect.height = roi->height + 2 * iradius;

  if (o->clip)
    {
      gegl_rectangle_intersect (
        &rect,
        &rect,
        gegl_operation_source_get_bounding_box (operation, "input"));
    }

  size = MIN (size, rect.height);

  in    = (gfloat *) gegl_malloc (4 * sizeof (gfloat) * rect.width * size);
  in_w  = (gfloat *) gegl_malloc (    sizeof (gfloat) * rect.width * size);
  out   = (gfloat *) gegl_malloc (4 * sizeof (gfloat) * roi->width);
  out_w = (gfloat *) gegl_malloc (    sizeof (gfloat) * roi->width);

  if (aux)
    mask = (gfloat *) gegl_malloc (sizeof (gfloat) * rect.width * size);

  auto row_index = [&] (gint y)
  {
    return (y - rect.y) % size;
  };

  auto weight = [&] (gfloat v)
  {
    v = (v                        - highlight_threshold_low) /
        (highlight_threshold_high - highlight_threshold_low);

    if (v <= 0.0f)
      return 1.0f;
    else if (v >= 1.0f)
      return highlight_max;

    return expf (v * highlight_factor);
  };

  auto read = [&] (gint y,
                   gint height)
  {
    gfloat *row;
    gfloat *row_w;
    gfloat *row_m = NULL;
    gint    n = rect.width * height;
    gint    r_i;
    gint    i;

    r_i = row_index (y);

    row   = in   + 4 * rect.width * r_i;
    row_w = in_w +     rect.width * r_i;

    gegl_buffer_get (input,
                     GEGL_RECTANGLE (rect.x, y, rect.width, height),
                     1.0,
                     format,
                     row,
                     GEGL_AUTO_ROWSTRIDE,
                     GEGL_ABYSS_NONE);

    if (mask)
      {
        row_m = mask + rect.width * r_i;

        gegl_buffer_get (aux,
                         GEGL_RECTANGLE (rect.x, y, rect.width, height),
                         1.0,
                         aux_format,
                         row_m,
                         GEGL_AUTO_ROWSTRIDE,
                         GEGL_ABYSS_NONE);
      }

    if (highlight_factor)
      {
        babl_process (weight_fish, row, row_w, n);
      }
    else
      {
        gfloat w = 1.0f;

        gegl_memset_pattern (row_w, &w, sizeof (gfloat), n);
      }

    for (i = 0; i < n; i++)
      {
        gint c;

        if (highlight_factor)
          row_w[i] = weight (row_w[i]);

        if (mask)
          {
            row_m[i]  = row_m[i] * radius + 0.5f;
            row_m[i] *= row_m[i];

            row_w[i] /= G_PI * row_m[i];
          }

        row[4 * i + 3] *= row_w[i];

        for (c = 0; c < 3; c++)
          row[4 * i + c] *= row[4 * i + 3];
      }
  };

  read (rect.y, MIN (roi->y + iradius + 1 - rect.y, rect.height));

  for (y = roi->y; y < roi->y + roi->height; y++)
    {
      gint r1, r2;
      gint x;

      memset (out,   0, 4 * sizeof (gfloat) * roi->width);
      memset (out_w, 0,     sizeof (gfloat) * roi->width);

      r1 = MAX (-iradius,                   - (y - rect.y));
      r2 = MIN (+iradius, (rect.height - 1) - (y - rect.y));

      if (! mask)
        {
          gint r;

          for (r = r1; r <= r2; r++)
            {
              gfloat       *o        = out;
              gfloat       *o_w      = out_w;
              const gfloat *row;
              const gfloat *row_w;
              gfloat        accum[4] = {};
              gfloat        accum_w  = 0.0f;
              gint          r_i;
              gint          s;
              gint          x1, x2;

              r_i = row_index (y + r);

              row   = in   + 4 * (rect.width * r_i + (roi->x - rect.x));
              row_w = in_w +     (rect.width * r_i + (roi->x - rect.x));

              s = sqrtf ((radius + 0.5f) * (radius + 0.5f) - r * r);

              x1 = MAX (-s,                  - (roi->x - rect.x));
              x2 = MIN (+s, (rect.width - 1) - (roi->x - rect.x));

              for (x = x1; x <= x2; x++)
                {
                  gint c;

                  for (c = 0; c < 4; c++)
                    accum[c] += row[4 * x + c];

                  accum_w += row_w[x];
                }

              x1 =                  - (roi->x - rect.x) + s;
              x2 = (rect.width - 1) - (roi->x - rect.x) - s;

              for (x = 0; x < roi->width; x++)
                {
                  gint c;

                  for (c = 0; c < 4; c++)
                    o[c] += accum[c];

                  *o_w += accum_w;

                  o += 4;
                  o_w++;

                  if (x >= x1)
                    {
                      for (c = 0; c < 4; c++)
                        accum[c] -= row[4 * (-s) + c];

                      accum_w -= row_w[-s];
                    }

                  row += 4;
                  row_w++;

                  if (x + 1 <= x2)
                    {
                      for (c = 0; c < 4; c++)
                        accum[c] += row[4 * (+s) + c];

                      accum_w += row_w[+s];
                    }
                }
            }
        }
      else
        {
          gint r;

          for (r = r1; r <= r2; r++)
            {
              const gfloat *row;
              const gfloat *row_w;
              const gfloat *row_m;
              gfloat        r2 = r * r;
              gint          r_i;
              gint          s;
              gint          x1, x2;

              r_i = row_index (y + r);

              row   = in   + 4 * (rect.width * r_i + (roi->x - rect.x));
              row_w = in_w +     (rect.width * r_i + (roi->x - rect.x));
              row_m = mask +     (rect.width * r_i + (roi->x - rect.x));

              s = sqrtf ((radius + 0.5f) * (radius + 0.5f) - r * r);

              x1 = MAX (                 - s,
                                         - (roi->x - rect.x));
              x2 = MIN ((roi->width - 1) + s,
                        (rect.width - 1) - (roi->x - rect.x));

              row   += 4 * x1;
              row_w +=     x1;
              row_m +=     x1;

              for (x = x1; x <= x2; x++, row += 4, row_w++, row_m++)
                {
                  gint xl, xr;
                  gint c;

                  if (*row_m < r2)
                    continue;

                  s = sqrtf (*row_m - r2);

                  xl = x - s;
                  xr = x + s + 1;

                  xl = MAX (xl, 0);

                  if (xl >= xr || xl >= roi->width)
                    continue;

                  for (c = 0; c < 4; c++)
                    out[4 * xl + c] += row[c];

                  out_w[xl] += *row_w;

                  if (xr < roi->width)
                    {
                      for (c = 0; c < 4; c++)
                        out[4 * xr + c] -= row[c];

                      out_w[xr] -= *row_w;
                    }
                }
            }

          for (x = 1; x < roi->width; x++)
            {
              gint c;

              for (c = 0; c < 4; c++)
                out[4 * x + c] += out[4 * (x - 1) + c];

              out_w[x] += out_w[x - 1];
            }
        }

      for (x = 0; x < roi->width; x++)
        {
          gint c;

          for (c = 0; c < 3; c++)
            out[4 * x + c] /= out[4 * x + 3];

          out[4 * x + 3] /= out_w[x];
        }

      gegl_buffer_set (output,
                       GEGL_RECTANGLE (roi->x,
                                       y,
                                       roi->width,
                                       1),
                       0,
                       format,
                       out,
                       GEGL_AUTO_ROWSTRIDE);

      if ((y + 1) + iradius < rect.y + rect.height)
        read ((y + 1) + iradius, 1);
    }

  gegl_free (mask);
  gegl_free (out_w);
  gegl_free (out);
  gegl_free (in_w);
  gegl_free (in);

  return TRUE;
}

static void
gegl_op_class_init (GeglOpClass *klass)
{
  GeglOperationClass         *operation_class;
  GeglOperationComposerClass *composer_class;

  operation_class = GEGL_OPERATION_CLASS (klass);
  composer_class  = GEGL_OPERATION_COMPOSER_CLASS (klass);

  operation_class->prepare                   = prepare;
  operation_class->get_bounding_box          = get_bounding_box;
  operation_class->get_required_for_output   = get_required_for_output;
  operation_class->get_invalidated_by_change = get_required_for_output;
  operation_class->process                   = operation_process;
  operation_class->opencl_support            = TRUE;

  composer_class->process                    = process;

  gegl_operation_class_set_keys (operation_class,
    "name",        "gegl:lens-blur",
    "title",       _("Lens Blur"),
    "categories",  "blur",
    "reference-hash", "c5dc4c97b0dacbe3fee41cefca1e6f42",
    "description", _("Simulate out-of-focus lens blur"),
    NULL);
}

#endif
