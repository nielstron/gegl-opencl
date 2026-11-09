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
 * Copyright 2020 Ell
 */

#include "config.h"
#include <glib/gi18n-lib.h>

/* #define MANUAL_CONTROL */

#define MAX_GAMMA 1000.0

#ifdef GEGL_PROPERTIES

enum_start (gegl_focus_blur_type)
  enum_value (GEGL_FOCUS_BLUR_TYPE_GAUSSIAN, "gaussian", N_("Gaussian Blur"))
  enum_value (GEGL_FOCUS_BLUR_TYPE_LENS,     "lens",     N_("Lens Blur"))
enum_end (GeglFocusBlurType)

enum_start (gegl_focus_blur_shape)
  enum_value (GEGL_FOCUS_BLUR_SHAPE_CIRCLE,     "circle",     N_("Circle"))
  enum_value (GEGL_FOCUS_BLUR_SHAPE_SQUARE,     "square",     N_("Square"))
  enum_value (GEGL_FOCUS_BLUR_SHAPE_DIAMOND,    "diamond",    N_("Diamond"))
  enum_value (GEGL_FOCUS_BLUR_SHAPE_HORIZONTAL, "horizontal", N_("Horizontal"))
  enum_value (GEGL_FOCUS_BLUR_SHAPE_VERTICAL,   "vertical",   N_("Vertical"))
enum_end (GeglFocusBlurShape)

property_enum (blur_type, _("Blur type"),
               GeglFocusBlurType, gegl_focus_blur_type,
               GEGL_FOCUS_BLUR_TYPE_GAUSSIAN)

property_double (blur_radius, _("Blur radius"), 25.0)
    description (_("Out-of-focus blur radius"))
    value_range (0.0, 1500.0)
    ui_range    (0.0, 100.0)
    ui_gamma    (2.0)
    ui_meta     ("unit", "pixel-distance")

property_double (highlight_factor, _("Highlight factor"), 0.0)
    description (_("Relative highlight strength"))
    value_range (0.0, 1.0)
    ui_meta     ("visible", "blur-type {lens}")

property_double (highlight_threshold_low, _("Highlight threshold (low)"), 0.0)
    ui_range    (0.0, 1.0)
    ui_meta     ("role", "range-start")
    ui_meta     ("unit", "luminance")
    ui_meta     ("range-label", _("Highlight threshold"))
    ui_meta     ("visible", "$highlight-factor.visible")

property_double (highlight_threshold_high, _("Highlight threshold (high)"), 1.0)
    ui_range    (0.0, 1.0)
    ui_meta     ("role", "range-end")
    ui_meta     ("unit", "luminance")
    ui_meta     ("visible", "$highlight-threshold-low.visible")

property_enum (shape, _("Shape"),
               GeglFocusBlurShape,
               gegl_focus_blur_shape,
               GEGL_FOCUS_BLUR_SHAPE_CIRCLE)

property_double (x, _("Center X"), 0.5)
    ui_range    (0, 1.0)
    ui_meta     ("unit", "relative-coordinate")
    ui_meta     ("axis", "x")

property_double (y, _("Center Y"), 0.5)
    ui_range    (0, 1.0)
    ui_meta     ("unit", "relative-coordinate")
    ui_meta     ("axis", "y")

property_double (radius, _("Radius"), 0.75)
    description (_("Focus-region outer radius"))
    value_range (0.0, G_MAXDOUBLE)
    ui_range    (0.0, 5.0)
    ui_meta     ("unit", "relative-distance")

property_double (focus, _("Sharpness"), 0.25)
    description (_("Focus-region inner limit"))
    value_range (0.0, 1.0)

property_double (midpoint, _("Midpoint"), 0.5)
    description (_("Focus-transition midpoint"))
    value_range (0.0, 1.0)

property_double (aspect_ratio, _("Aspect ratio"), 0.0)
    value_range (-1.0, +1.0)

property_double (rotation, _("Rotation"), 0.0)
    value_range (-180.0, +180.0)
    ui_meta     ("unit", "degree")
    ui_meta     ("direction", "cw")

#ifdef MANUAL_CONTROL

property_int (blur_levels, _("Blur levels"), 8)
    description (_("Number of blur levels"))
    value_range (2, 16)
    ui_meta     ("visible", "blur-type {gaussian}")

property_double (blur_gamma, _("Blur gamma"), 1.5)
    description (_("Gamma factor for blur-level spacing"))
    value_range (0.0, G_MAXDOUBLE)
    ui_range    (0.1, 10.0)
    ui_meta     ("visible", "blur-type {gaussian}")

#else

property_boolean (high_quality, _("High quality"), FALSE)
    description (_("Generate more accurate and consistent output (slower)"))
    ui_meta     ("visible", "blur-type {gaussian}")

#endif

#else

#define GEGL_OP_FILTER
#define GEGL_OP_NAME     focus_blur
#define GEGL_OP_C_SOURCE focus-blur.c

#include "gegl-op.h"
#include "process/gegl-eval-manager-private.h"
#include "opencl/gblur-mem-cl.h"
#include "opencl/focus-blur-mask.h"
#include "opencl/focus-lens-cl.h"
#include "opencl/variable-blur.cl.h"
#include "variable-blur-params.h"

static void
get_mask_parameters (GeglProperties *o,
                     gdouble        *gamma,
                     gdouble        *squeeze,
                     gdouble        *rotation)
{
  gdouble scale;

  if (o->aspect_ratio >= 0.0)
    scale = 1.0 - o->aspect_ratio;
  else
    scale = 1.0 / (1.0 + o->aspect_ratio);

  if (scale <= 1.0)
    *squeeze = +2.0 * atan (1.0 / scale - 1.0) / G_PI;
  else
    *squeeze = -2.0 * atan (scale - 1.0) / G_PI;

  *gamma = o->midpoint < 1.0 ?
    MIN (log (0.5) / log (o->midpoint), MAX_GAMMA) : MAX_GAMMA;
  *rotation = fmod (o->rotation + 360.0, 360.0);
}

/* Keep the original graph as the CPU implementation. */
static GeglBuffer *
process_graph (GeglProperties      *o,
               GeglBuffer          *source,
               const GeglRectangle *result,
               gint                 level)
{
  GeglNode        *graph = gegl_node_new ();
  GeglNode        *input, *color, *crop, *vignette, *blur;
  GeglColor       *black = gegl_color_new ("black");
  GeglColor       *white = gegl_color_new ("white");
  GeglEvalManager *eval;
  GeglBuffer      *output;
  gdouble          gamma, squeeze, rotation;

  get_mask_parameters (o, &gamma, &squeeze, &rotation);
  g_object_set (graph, "use-opencl", FALSE, NULL);

  input = gegl_node_new_child (graph,
                               "operation", "gegl:buffer-source",
                               "buffer", source,
                               NULL);
  color = gegl_node_new_child (graph,
                               "operation", "gegl:color",
                               "value", black,
                               NULL);
  crop = gegl_node_new_child (graph, "operation", "gegl:crop", NULL);
  vignette = gegl_node_new_child (graph,
                                  "operation", "gegl:vignette",
                                  "color", white,
                                  "proportion", 0.0,
                                  "shape", o->shape,
                                  "radius", o->radius,
                                  "softness", 1.0 - o->focus,
                                  "gamma", gamma,
                                  "squeeze", squeeze,
                                  "x", o->x,
                                  "y", o->y,
                                  "rotation", rotation,
                                  NULL);

  if (o->blur_type == GEGL_FOCUS_BLUR_TYPE_GAUSSIAN)
    blur = gegl_node_new_child (graph,
                                "operation", "gegl:variable-blur",
                                "radius", o->blur_radius,
                                "linear-mask", TRUE,
#ifdef MANUAL_CONTROL
                                "levels", o->blur_levels,
                                "gamma", o->blur_gamma,
#else
                                "high-quality", o->high_quality,
#endif
                                NULL);
  else
    blur = gegl_node_new_child (graph,
                                "operation", "gegl:lens-blur",
                                "radius", o->blur_radius,
                                "linear-mask", TRUE,
                                "highlight-factor", o->highlight_factor,
                                "highlight-threshold-low",
                                o->highlight_threshold_low,
                                "highlight-threshold-high",
                                o->highlight_threshold_high,
                                NULL);

  gegl_node_link (input, blur);
  gegl_node_link_many (color, crop, vignette, NULL);
  gegl_node_connect (input, "output", crop, "aux");
  gegl_node_connect (vignette, "output", blur, "aux");

  eval = gegl_eval_manager_new (blur, "output");
  output = gegl_eval_manager_apply (eval, result, level);

  g_object_unref (eval);
  g_object_unref (white);
  g_object_unref (black);
  g_object_unref (graph);

  return output;
}

static void
prepare (GeglOperation *operation)
{
  GeglProperties *o = GEGL_PROPERTIES (operation);
  const Babl     *space = gegl_operation_get_source_space (operation, "input");
  const gchar    *encoding = o->blur_type == GEGL_FOCUS_BLUR_TYPE_LENS ?
    "RGBA float" : "RaGaBaA float";

  gegl_operation_set_format (operation, "input",
                             gegl_operation_get_source_format (operation, "input"));
  gegl_operation_set_format (operation, "output",
                             babl_format_with_space (encoding, space));
}

static GeglRectangle
get_required_for_output (GeglOperation       *operation,
                         const gchar         *input_pad,
                         const GeglRectangle *roi)
{
  GeglProperties      *o = GEGL_PROPERTIES (operation);
  const GeglRectangle *extent =
    gegl_operation_source_get_bounding_box (operation, "input");
  GeglRectangle        required = *roi;

  if (o->blur_type == GEGL_FOCUS_BLUR_TYPE_LENS)
    {
      gint radius = floor (o->blur_radius + 0.5);

      required.x      -= radius;
      required.y      -= radius;
      required.width  += 2 * radius;
      required.height += 2 * radius;
    }
  else if (extent && ! gegl_rectangle_is_infinite_plane (extent))
    {
      /* IIR filtering propagates along whole rows and columns. */
      required = *extent;
    }
  else if (extent)
    {
      /* On an infinite plane the IIR passes are passthrough. FIR passes
       * have sigma below one and use at most seven taps per axis. */
      required.x      -= 3;
      required.y      -= 3;
      required.width  += 6;
      required.height += 6;
    }

  return required;
}

static GeglRectangle
get_invalidated_by_change (GeglOperation       *operation,
                           const gchar         *input_pad,
                           const GeglRectangle *roi)
{
  return get_required_for_output (operation, input_pad, roi);
}

static gboolean
write_gaussian_result (GeglBuffer          *output,
                       const GeglRectangle *result,
                       const Babl          *format,
                       cl_mem               pixels)
{
  const size_t          bpp = 4 * sizeof (cl_float);
  GeglBufferClIterator *i = gegl_buffer_cl_iterator_new (output, result, format,
                                                         GEGL_CL_BUFFER_WRITE);
  gboolean              error = FALSE;

  while (gegl_buffer_cl_iterator_next (i, &error) && ! error)
    {
      const GeglRectangle *roi = &i->roi[0];
      const size_t src_origin[3] = { (roi->x - result->x) * bpp,
                                      roi->y - result->y, 0 };
      const size_t dst_origin[3] = { 0, 0, 0 };
      const size_t region[3] = { roi->width * bpp, roi->height, 1 };

      if (gegl_clEnqueueCopyBufferRect (gegl_cl_get_command_queue (),
                                         pixels, i->tex[0],
                                         src_origin, dst_origin, region,
                                         result->width * bpp, 0, region[0], 0,
                                         0, NULL, NULL) != CL_SUCCESS)
        {
          gegl_buffer_cl_iterator_stop (i);
          return FALSE;
        }
    }

  return ! error;
}

static gboolean
gaussian_cl_process (GeglProperties      *o,
                     GeglBuffer          *source,
                     GeglBuffer          *output,
                     const GeglRectangle *extent,
                     const GeglRectangle *result,
                     const Babl          *format,
                     const FocusBlurMask *mask)
{
#if defined (__aarch64__) || defined (__FMA__)
  const gchar *options = "#define VARIABLE_BLUR_USE_FMA 1\n";
#else
  const gchar *options = "";
#endif
  static GeglClRunData *cl_data;
  cl_mem               image = NULL, blur = NULL, horizontal = NULL;
  cl_mem               causal = NULL, values = NULL, pixels = NULL;
  const size_t         global_ws[2] = { result->width, result->height };
  const gsize          n = (gsize) extent->width * extent->height;
  const gsize          result_n = (gsize) result->width * result->height;
  const gsize          image_bytes = n * 4 * sizeof (cl_float);
  const cl_int         stride = extent->width;
  const cl_int         offset = (result->y - extent->y) * (gint64) stride +
                                result->x - extent->x;
  const Babl          *source_format = gegl_buffer_get_format (source);
  gboolean             segment_lines = TRUE;
  cl_float             edges[VARIABLE_BLUR_MAX_LEVELS];
  gdouble              gamma;
  cl_int               levels;
  cl_int               error;
  gboolean             success = FALSE;

  if (! cl_data)
    {
      const char *names[] = { "variable_blur_blend", NULL };
      gchar *kernel_source = g_strconcat (options, variable_blur_cl_source, NULL);

      cl_data = gegl_cl_compile_and_build_optimized (kernel_source, names);
      g_free (kernel_source);
    }

  if (! cl_data || ! gblur_mem_cl_build ())
    return FALSE;

  /* The finite warmup of an independent IIR segment can miss a NaN from
   * earlier in the line. Use the full recurrence for floating-point input. */
  for (gint i = 0; i < babl_format_get_n_components (source_format); i++)
    {
      const Babl *type = babl_format_get_type (source_format, i);

      if (type != babl_type ("u8") && type != babl_type ("u16") &&
          type != babl_type ("u32"))
        segment_lines = FALSE;
    }

#ifdef MANUAL_CONTROL
  levels = o->blur_levels;
  gamma  = o->blur_gamma;
#else
  levels = variable_blur_levels (o->blur_radius, o->high_quality);
  gamma  = VARIABLE_BLUR_GAMMA;
#endif

  /* With two levels piecewise-blend uses a linear ramp. The blur
   * radius still follows variable-blur's spacing. */
  for (gint i = 0; i < levels; i++)
    edges[i] = powf (i * (1.0f / (levels - 1)), levels == 2 ? 1.0f : gamma);

  image = gegl_clCreateBuffer (gegl_cl_get_context (), CL_MEM_READ_WRITE,
                               image_bytes, NULL, &error);
  if (error != CL_SUCCESS)
    goto cleanup;
  blur = gegl_clCreateBuffer (gegl_cl_get_context (), CL_MEM_READ_WRITE,
                              image_bytes, NULL, &error);
  if (error != CL_SUCCESS)
    goto cleanup;
  horizontal = gegl_clCreateBuffer (gegl_cl_get_context (), CL_MEM_READ_WRITE,
                                    image_bytes, NULL, &error);
  if (error != CL_SUCCESS)
    goto cleanup;
  if (segment_lines && (gfloat) o->blur_radius >= 1.0f)
    {
      causal = gegl_clCreateBuffer (gegl_cl_get_context (), CL_MEM_READ_WRITE,
                                    image_bytes, NULL, &error);
      if (error != CL_SUCCESS)
        goto cleanup;
    }
  values = gegl_clCreateBuffer (gegl_cl_get_context (), CL_MEM_READ_WRITE,
                                result_n * sizeof (cl_float), NULL, &error);
  if (error != CL_SUCCESS)
    goto cleanup;
  pixels = gegl_clCreateBuffer (gegl_cl_get_context (), CL_MEM_READ_WRITE,
                                result_n * 4 * sizeof (cl_float), NULL, &error);
  if (error != CL_SUCCESS)
    goto cleanup;

  if (! focus_blur_mask_enqueue (mask, result, values) ||
      ! gblur_mem_cl_read (source, extent, format, image, FALSE))
    goto cleanup;

  for (cl_int i = 0; i < levels; i++)
    {
      const cl_mem level = i ? blur : image;
      const cl_float edge = edges[i];
      const cl_float previous = i ? edges[i - 1] : 0.0f;
      const cl_float next = i + 1 < levels ? edges[i + 1] : 1.0f;
      const cl_float inv_range = i ? 1.0f / (edge - previous) : 0.0f;

      if (i)
        {
          gfloat sigma = variable_blur_sigma (o->blur_radius, i, levels, gamma);

          if (! gblur_mem_cl_pass (image, horizontal, causal,
                                   extent->width, extent->height, 4,
                                   sigma, TRUE, TRUE, segment_lines) ||
              ! gblur_mem_cl_pass (horizontal, blur, causal,
                                   extent->width, extent->height, 4,
                                   sigma, FALSE, TRUE, segment_lines))
            goto cleanup;
        }

      error = gegl_cl_set_kernel_args (cl_data->kernel[0],
                                       sizeof (cl_mem), &values,
                                       sizeof (cl_mem), &level,
                                       sizeof (cl_mem), &pixels,
                                       sizeof (cl_int), &stride,
                                       sizeof (cl_int), &offset,
                                       sizeof (cl_float), &previous,
                                       sizeof (cl_float), &edge,
                                       sizeof (cl_float), &next,
                                       sizeof (cl_float), &inv_range,
                                       sizeof (cl_int), &i,
                                       sizeof (cl_int), &levels,
                                       NULL);
      if (error != CL_SUCCESS ||
          gegl_clEnqueueNDRangeKernel (gegl_cl_get_command_queue (),
                                        cl_data->kernel[0], 2, NULL,
                                        global_ws, NULL, 0, NULL, NULL) != CL_SUCCESS)
        goto cleanup;
    }

  success = write_gaussian_result (output, result, format, pixels);

cleanup:
  if (pixels)     gegl_clReleaseMemObject (pixels);
  if (values)     gegl_clReleaseMemObject (values);
  if (causal)     gegl_clReleaseMemObject (causal);
  if (horizontal) gegl_clReleaseMemObject (horizontal);
  if (blur)       gegl_clReleaseMemObject (blur);
  if (image)      gegl_clReleaseMemObject (image);

  return success;
}

static gboolean
operation_process (GeglOperation        *operation,
                   GeglOperationContext *context,
                   const gchar          *output_pad,
                   const GeglRectangle  *result,
                   gint                  level)
{
  GeglProperties      *o = GEGL_PROPERTIES (operation);
  GeglBuffer          *input = GEGL_BUFFER (
    gegl_operation_context_get_object (context, "input"));
  const GeglRectangle *extent =
    gegl_operation_source_get_bounding_box (operation, "input");
  gboolean             finite = extent && ! gegl_rectangle_is_infinite_plane (extent);
  GeglBuffer          *source = input;
  GeglBuffer          *output = NULL;
  gboolean             success = FALSE;

  if (extent && ! gegl_rectangle_equal (gegl_buffer_get_extent (input), extent))
    source = gegl_buffer_create_sub_buffer (input, extent);

  if (o->blur_type == GEGL_FOCUS_BLUR_TYPE_LENS && o->blur_radius < 0.5)
    {
      gegl_operation_context_set_object (context, "output", G_OBJECT (source));
      success = TRUE;
    }
  else if (level == 0 && finite &&
           gegl_operation_use_opencl (operation) &&
           gegl_rectangle_contains (extent, result) &&
           (gint64) extent->width * extent->height <= G_MAXINT / 4)
    {
      FocusBlurMask mask;
      gdouble       gamma, squeeze, rotation;

      get_mask_parameters (o, &gamma, &squeeze, &rotation);

      if (focus_blur_mask_init (&mask, extent, o->shape, o->radius,
                                1.0 - o->focus, gamma, squeeze, o->x, o->y,
                                rotation))
        {
          const Babl *format = gegl_operation_get_format (operation, "output");

          if (o->blur_type == GEGL_FOCUS_BLUR_TYPE_LENS)
            {
              output = gegl_operation_context_get_target (context, "output");
              success = focus_lens_cl_process (source, output, extent, result,
                                                format, &mask, o->blur_radius,
                                                o->highlight_factor,
                                                o->highlight_threshold_low,
                                                o->highlight_threshold_high);
            }
          else if (gegl_rectangle_equal (gegl_buffer_get_abyss (source), extent) &&
                   !(babl_get_model_flags (gegl_buffer_get_format (source)) &
                     BABL_MODEL_FLAG_CMYK))
            {
              output = gegl_operation_context_get_target (context, "output");
              success = gaussian_cl_process (o, source, output, extent, result,
                                             format, &mask);
            }
        }
    }

  if (! success)
    {
      output = process_graph (o, source, result, level);
      if (output)
        gegl_operation_context_take_object (context, "output", G_OBJECT (output));
      success = output != NULL;
    }

  if (source != input)
    g_object_unref (source);

  return success;
}

static void
gegl_op_class_init (GeglOpClass *klass)
{
  GeglOperationClass *operation_class = GEGL_OPERATION_CLASS (klass);

  operation_class->prepare                   = prepare;
  operation_class->get_required_for_output   = get_required_for_output;
  operation_class->get_invalidated_by_change = get_invalidated_by_change;
  operation_class->threaded                  = FALSE;
  operation_class->opencl_support            = TRUE;
  operation_class->process                   = operation_process;

  gegl_operation_class_set_keys (operation_class,
    "name",        "gegl:focus-blur",
    "title",       _("Focus Blur"),
    "categories",  "blur",
    "reference-hash", "a6f7a6425769c7d8b1d277a5c3f25973",
    "description", _("Blur the image around a focal point"),
    NULL);
}

#endif
