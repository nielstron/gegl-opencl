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
 * This operation is inspired from the color enhance gimp plugin.
 * It alters the chroma component to cover maximum possible range,
 * keeping hue and lightness untouched.
 *
 * Thomas Manni <thomas.manni@free.fr>
 */

#include "config.h"
#include <glib/gi18n-lib.h>

#ifdef GEGL_PROPERTIES

#else

#define GEGL_OP_FILTER
#define GEGL_OP_NAME     color_enhance
#define GEGL_OP_C_SOURCE color-enhance.c

#include "gegl-op.h"

/* Achromatic input has a chroma range made only of the rounding noise of
 * the CIE LCH(ab) conversion (about 1e-4), which stretching would amplify
 * into random colors. Leave input with a smaller range unchanged.
 */
#define MIN_CHROMA_RANGE 0.01

static void
buffer_get_min_max (GeglOperation       *operation,
                    GeglBuffer          *buffer,
                    const GeglRectangle *result,
                    gdouble             *min,
                    gdouble             *max,
                    const Babl          *format)
{
  GeglBufferIterator *gi;
  gint                done_pixels = 0;

  gegl_operation_progress (operation, 0.0, "");

  gi = gegl_buffer_iterator_new (buffer, result, 0, format,
                                 GEGL_ACCESS_READ, GEGL_ABYSS_NONE, 1);

  *min = G_MAXDOUBLE;
  *max = -G_MAXDOUBLE;

  while (gegl_buffer_iterator_next (gi))
    {
      gint o;
      gfloat *buf = gi->items[0].data;

      for (o = 0; o < gi->length; o++)
        {
          *min = MIN (buf[1], *min);
          *max = MAX (buf[1], *max);
          buf += 3;
        }

      done_pixels += gi->length;

      gegl_operation_progress (operation,
                               (gdouble) 0.5 * done_pixels /
                               (gdouble) (result->width * result->height),
                               "");
    }

  gegl_operation_progress (operation, 0.5, "");
}

/* The CIE LCH(ab) format the CPU path works in */
static const Babl *
get_lch_format (GeglOperation *operation)
{
  const Babl *space     = gegl_operation_get_source_space (operation, "input");
  const Babl *in_format = gegl_operation_get_source_format (operation, "input");

  if (in_format && babl_format_has_alpha (in_format))
    return babl_format_with_space ("CIE LCH(ab) alpha float", space);

  return babl_format_with_space ("CIE LCH(ab) float", space);
}

static void prepare (GeglOperation *operation)
{
  const Babl *space = gegl_operation_get_source_space (operation, "input");
  const Babl *format;

  /* The OpenCL path converts to and from CIE LCH(ab) itself. For sRGB it
   * also applies the TRC, since R'G'B'A float is cheaper to convert to the
   * usual 8-bit formats on the CPU.
   */
  if (gegl_operation_use_opencl (operation))
    {
      format = babl_format_with_space ("RGBA float", space);
      gegl_operation_set_format (operation, "input", format);

      if (babl_format_get_space (format) == babl_space ("sRGB"))
        format = babl_format ("R'G'B'A float");

      gegl_operation_set_format (operation, "output", format);
      return;
    }

  format = get_lch_format (operation);

  gegl_operation_set_format (operation, "input", format);
  gegl_operation_set_format (operation, "output", format);
}

static GeglRectangle
get_required_for_output (GeglOperation       *operation,
                         const gchar         *input_pad,
                         const GeglRectangle *roi)
{
  GeglRectangle result = *gegl_operation_source_get_bounding_box (operation, "input");

  /* Don't request an infinite plane */
  if (gegl_rectangle_is_infinite_plane (&result))
    return *roi;

  return result;
}

static GeglRectangle
get_cached_region (GeglOperation       *operation,
                   const GeglRectangle *roi)
{
  GeglRectangle result = *gegl_operation_source_get_bounding_box (operation, "input");

  if (gegl_rectangle_is_infinite_plane (&result))
    return *roi;

  return result;
}

#include "opencl/gegl-cl.h"
#include "gegl-buffer-cl-iterator.h"
#include "opencl/color-enhance.cl.h"

static GeglClRunData *cl_data = NULL;

/* Rows of the RGB to XYZ (D50) matrix of space, and of its inverse */
static void
cl_get_matrices (const Babl *space,
                 cl_float4   rgb_to_xyz[3],
                 cl_float4   xyz_to_rgb[3])
{
  const gfloat unit[9] = { 1.0f, 0.0f, 0.0f,
                           0.0f, 1.0f, 0.0f,
                           0.0f, 0.0f, 1.0f };
  gfloat       cols[9];
  gint         r, c;

  babl_process (babl_fish (babl_format_with_space ("RGB float", space),
                           babl_format_with_space ("CIE XYZ float", space)),
                unit, cols, 3);

  for (r = 0; r < 3; r++)
    {
      for (c = 0; c < 3; c++)
        rgb_to_xyz[r].s[c] = cols[c * 3 + r];
      rgb_to_xyz[r].s[3] = 0.0f;
    }

  babl_process (babl_fish (babl_format_with_space ("CIE XYZ float", space),
                           babl_format_with_space ("RGB float", space)),
                unit, cols, 3);

  for (r = 0; r < 3; r++)
    {
      for (c = 0; c < 3; c++)
        xyz_to_rgb[r].s[c] = cols[c * 3 + r];
      xyz_to_rgb[r].s[3] = 0.0f;
    }
}

/* Folds the (min chroma, -max chroma) of in_tex into acc */
static gboolean
cl_get_min_max (cl_mem           in_tex,
                size_t           n_pixels,
                const cl_float4  m[3],
                gfloat           acc[2])
{
  cl_int     cl_err   = 0;
  cl_mem     cl_part  = NULL;
  cl_float2  part[256];
  cl_int     n        = n_pixels;
  size_t     local_ws = 256;
  size_t     max_local_ws;
  size_t     work_groups;
  size_t     global_ws;
  gint       i;

  if (n_pixels < 1)
    return FALSE;

  cl_err = gegl_clGetDeviceInfo (gegl_cl_get_device (),
                                 CL_DEVICE_MAX_WORK_GROUP_SIZE,
                                 sizeof (size_t), &max_local_ws, NULL);
  CL_CHECK;

  max_local_ws = MIN (max_local_ws, cl_data->work_group_size[0]);

  /* Needs to be a power of two */
  while (local_ws > max_local_ws)
    local_ws /= 2;

  work_groups = MIN ((n_pixels + local_ws - 1) / local_ws, 256);
  global_ws   = work_groups * local_ws;

  cl_part = gegl_clCreateBuffer (gegl_cl_get_context (),
                                 CL_MEM_WRITE_ONLY,
                                 work_groups * sizeof (cl_float2),
                                 NULL, &cl_err);
  CL_CHECK;

  cl_err = gegl_cl_set_kernel_args (cl_data->kernel[0],
                                    sizeof (cl_mem),    &in_tex,
                                    sizeof (cl_mem),    &cl_part,
                                    local_ws * sizeof (cl_float2), NULL,
                                    sizeof (cl_int),    &n,
                                    sizeof (cl_float4), &m[0],
                                    sizeof (cl_float4), &m[1],
                                    sizeof (cl_float4), &m[2],
                                    NULL);
  CL_CHECK;

  cl_err = gegl_clEnqueueNDRangeKernel (gegl_cl_get_command_queue (),
                                        cl_data->kernel[0], 1,
                                        NULL, &global_ws, &local_ws,
                                        0, NULL, NULL);
  CL_CHECK;

  cl_err = gegl_clEnqueueReadBuffer (gegl_cl_get_command_queue (),
                                     cl_part, CL_TRUE, 0,
                                     work_groups * sizeof (cl_float2), part,
                                     0, NULL, NULL);
  CL_CHECK;

  for (i = 0; i < work_groups; i++)
    {
      acc[0] = MIN (acc[0], part[i].s[0]);
      acc[1] = MIN (acc[1], part[i].s[1]);
    }

  cl_err = gegl_clReleaseMemObject (cl_part);
  CL_CHECK_ONLY (cl_err);

  return FALSE;

error:
  if (cl_part)
    gegl_clReleaseMemObject (cl_part);

  return TRUE;
}

static gboolean
cl_process (GeglOperation       *operation,
            GeglBuffer          *input,
            GeglBuffer          *output,
            const GeglRectangle *result)
{
  const Babl           *format     = gegl_operation_get_format (operation, "input");
  const Babl           *out_format = gegl_operation_get_format (operation, "output");
  GeglBufferClIterator *i;
  cl_float4             m[3], mi[3];
  cl_int                to_gamma;
  gfloat                acc[2] = { G_MAXFLOAT, G_MAXFLOAT };
  cl_float              min_chroma, delta;
  cl_int                cl_err = 0;
  gboolean              err    = FALSE;
  gint                  read;

  /* The kernels expect linear RGBA, and write RGBA or R'G'B'A (sRGB
   * only), which prepare () only selects when OpenCL is enabled.
   */
  if (!babl_model_is (babl_format_get_model (format), "RGBA"))
    return FALSE;

  if (babl_model_is (babl_format_get_model (out_format), "RGBA"))
    to_gamma = FALSE;
  else if (babl_format_get_space (out_format) == babl_space ("sRGB") &&
           babl_model_is (babl_format_get_model (out_format), "R'G'B'A"))
    to_gamma = TRUE;
  else
    return FALSE;

  if (!cl_data)
    {
      const char *kernel_name[] = { "chroma_min_max",
                                    "color_enhance",
                                    NULL };
      cl_data = gegl_cl_compile_and_build (color_enhance_cl_source,
                                           kernel_name);
    }

  if (!cl_data)
    return FALSE;

  cl_get_matrices (babl_format_get_space (format), m, mi);

  i = gegl_buffer_cl_iterator_new (input, result, format,
                                   GEGL_CL_BUFFER_READ);

  while (gegl_buffer_cl_iterator_next (i, &err) && !err)
    {
      err = cl_get_min_max (i->tex[0], i->size[0], m, acc);

      if (err)
        {
          gegl_buffer_cl_iterator_stop (i);
          break;
        }
    }

  if (err)
    return FALSE;

  min_chroma = acc[0];
  delta      = -acc[1] - acc[0];

  if (! delta)
    {
      gegl_buffer_copy (input, NULL, GEGL_ABYSS_NONE, output, NULL);
      return TRUE;
    }

  i = gegl_buffer_cl_iterator_new (output, result, out_format,
                                   GEGL_CL_BUFFER_WRITE);

  read = gegl_buffer_cl_iterator_add (i, input, result, format,
                                      GEGL_CL_BUFFER_READ, GEGL_ABYSS_NONE);

  while (gegl_buffer_cl_iterator_next (i, &err) && !err)
    {
      cl_err = gegl_cl_set_kernel_args (cl_data->kernel[1],
                                        sizeof (cl_mem),    &i->tex[read],
                                        sizeof (cl_mem),    &i->tex[0],
                                        sizeof (cl_float),  &min_chroma,
                                        sizeof (cl_float),  &delta,
                                        sizeof (cl_float4), &m[0],
                                        sizeof (cl_float4), &m[1],
                                        sizeof (cl_float4), &m[2],
                                        sizeof (cl_float4), &mi[0],
                                        sizeof (cl_float4), &mi[1],
                                        sizeof (cl_float4), &mi[2],
                                        sizeof (cl_int),    &to_gamma,
                                        NULL);
      CL_CHECK;

      cl_err = gegl_clEnqueueNDRangeKernel (gegl_cl_get_command_queue (),
                                            cl_data->kernel[1], 1,
                                            NULL, &i->size[0], NULL,
                                            0, NULL, NULL);
      CL_CHECK;
    }

  if (err)
    return FALSE;

  gegl_operation_progress (operation, 1.0, "");

  return TRUE;

error:
  gegl_buffer_cl_iterator_stop (i);

  return FALSE;
}

static gboolean
process (GeglOperation       *operation,
         GeglBuffer          *input,
         GeglBuffer          *output,
         const GeglRectangle *result,
         gint                 level)
{
  const Babl *format = gegl_operation_get_format (operation, "output");
  gboolean has_alpha;
  GeglBufferIterator *gi;
  gint                done_pixels = 0;
  gdouble  min;
  gdouble  max;
  gdouble  delta;

  if (gegl_operation_use_opencl (operation))
    {
      if (cl_process (operation, input, output, result))
        return TRUE;

      /* prepare () selected RGB formats for the OpenCL path, the CPU
       * path needs to work in CIE LCH(ab)
       */
      format = get_lch_format (operation);
    }

  has_alpha = babl_format_has_alpha (format);

  buffer_get_min_max (operation, input, result, &min, &max,
                      babl_format_with_space ("CIE LCH(ab) float",
                      babl_format_get_space (format)));

  gegl_operation_progress (operation, 0.5, "");

  delta = max - min;

  if (delta < MIN_CHROMA_RANGE)
    {
      gegl_buffer_copy (input, NULL, GEGL_ABYSS_NONE,
                        output, NULL);
      return TRUE;
    }

  gi = gegl_buffer_iterator_new (input, result, 0, format,
                                 GEGL_ACCESS_READ, GEGL_ABYSS_NONE, 2);

  gegl_buffer_iterator_add (gi, output, result, 0, format,
                            GEGL_ACCESS_WRITE, GEGL_ABYSS_NONE);


  if (has_alpha)
    {
      while (gegl_buffer_iterator_next (gi))
        {
          gfloat *in  = gi->items[0].data;
          gfloat *out = gi->items[1].data;

          gint i;
          for (i = 0; i < gi->length; i++)
            {
              out[0] = in[0];
              out[1] = (in[1] - min) / delta * 100.0;
              out[2] = in[2];
              out[3] = in[3];

              in  += 4;
              out += 4;
            }

          done_pixels += gi->length;

          gegl_operation_progress (operation,
                                   0.5 +
                                   (gdouble) 0.5 * done_pixels /
                                   (gdouble) (result->width * result->height),
                                   "");
       }
    }
  else
    {
       while (gegl_buffer_iterator_next (gi))
        {
          gfloat *in  = gi->items[0].data;
          gfloat *out = gi->items[1].data;

          gint i;
          for (i = 0; i < gi->length; i++)
            {
              out[0] = in[0];
              out[1] = (in[1] - min) / delta * 100.0;
              out[2] = in[2];

              in  += 3;
              out += 3;
            }

          done_pixels += gi->length;

          gegl_operation_progress (operation,
                                   0.5 +
                                   (gdouble) 0.5 * done_pixels /
                                   (gdouble) (result->width * result->height),
                                   "");
        }
    }

  gegl_operation_progress (operation, 1.0, "");

  return TRUE;
}

/* Pass-through when trying to perform a reduction on an infinite plane
 */
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

  operation_class = GEGL_OPERATION_CLASS (gegl_op_parent_class);

  if (in_rect && gegl_rectangle_is_infinite_plane (in_rect))
    {
      gpointer in = gegl_operation_context_get_object (context, "input");
      gegl_operation_context_take_object (context, "output",
                                          g_object_ref (G_OBJECT (in)));
      return TRUE;
    }

  /* chain up, which will create the needed buffers for our actual
   * process function
   */
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

  filter_class->process = process;

  operation_class->prepare = prepare;
  operation_class->process = operation_process;
  operation_class->get_required_for_output = get_required_for_output;
  operation_class->get_cached_region = get_cached_region;
  operation_class->opencl_support = TRUE;
  operation_class->threaded = FALSE;

  gegl_operation_class_set_keys (operation_class,
    "name",        "gegl:color-enhance",
    "title",       _("Color Enhance"),
    "categories" , "color:enhance",
    "reference-hash", "41ef79339fd722b64aa3520c33ff6f52",
    "description",
        _("Stretch color chroma to cover maximum possible range, "
          "keeping hue and lightness untouched."),
        NULL);
}

#endif
