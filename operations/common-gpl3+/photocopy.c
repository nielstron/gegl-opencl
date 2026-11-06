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
 * Copyright 2001 Spencer Kimball, Bit Specialists, Inc.
 * Copyright 2011 Hans Lo <hansshulo@gmail.com>
 */


#include "config.h"
#include <glib/gi18n-lib.h>

#ifdef GEGL_PROPERTIES

property_double (mask_radius, _("Mask radius"), 10.0)
    value_range (0.0, 50.0)

property_double (sharpness, _("Sharpness"), 0.5)
    value_range (0.0, 1.0)

property_double (black, _("Percent black"), 0.2)
    value_range (0.0, 1.0)

property_double (white, _("Percent white"), 0.2)
    value_range (0.0, 1.0)

#else

#define GEGL_OP_FILTER
#define GEGL_OP_NAME     photocopy
#define GEGL_OP_C_SOURCE photocopy.c

#include "gegl-op.h"

#define THRESHOLD 0.75

static void
grey_blur_buffer (GeglBuffer  *input,
                  gdouble      sharpness,
                  gdouble      mask_radius,
                  GeglBuffer **dest1,
                  GeglBuffer **dest2)
{
  GeglNode *gegl, *image, *write1, *write2, *blur1, *blur2;
  gdouble radius, std_dev1, std_dev2;

  gegl = gegl_node_new ();
  image = gegl_node_new_child (gegl,
                               "operation", "gegl:buffer-source",
                               "buffer", input,
                               NULL);

  radius   = MAX (1.0, 10 * (1.0 - sharpness));
  radius   = fabs (radius) + 1.0;
  std_dev1 = sqrt (-(radius * radius) / (2 * log (1.0 / 255.0)));

  radius   = fabs (mask_radius) + 1.0;
  std_dev2 = sqrt (-(radius * radius) / (2 * log (1.0 / 255.0)));

  blur1 =  gegl_node_new_child (gegl,
                                "operation", "gegl:gaussian-blur",
                                "std_dev_x", std_dev1,
                                "std_dev_y", std_dev1,
                                NULL);
  blur2 =  gegl_node_new_child (gegl,
                                "operation", "gegl:gaussian-blur",
                                "std_dev_x", std_dev2,
                                "std_dev_y", std_dev2,
                                NULL);

  write1 = gegl_node_new_child (gegl,
                                "operation", "gegl:buffer-sink",
                                "buffer", dest1, NULL);

  write2 = gegl_node_new_child (gegl,
                                "operation", "gegl:buffer-sink",
                                "buffer", dest2, NULL);

  gegl_node_link_many (image, blur1, write1, NULL);
  gegl_node_process (write1);

  gegl_node_link_many (image, blur2, write2, NULL);
  gegl_node_process (write2);

  g_object_unref (gegl);
}

static gdouble
calculate_threshold (gint    *hist,
                     gdouble  pct,
                     gint     count,
                     gint     under_threshold)
{
  gint    sum;
  gint    i;

  if (pct == 0.0 || count == 0)
    return (under_threshold ? 1.0 : 0.0);

  sum = 0;
  for (i = 0; i < 2000; i++)
    {
      sum += hist[i];
      if (((gdouble) sum / (gdouble) count) > pct)
        {
          if (under_threshold)
            return (THRESHOLD - (gdouble) i / 1000.0);
          else
            return ((gdouble) i / 1000.0 - THRESHOLD);
        }
    }

  return (under_threshold ? 0.0 : 1.0);
}

static void
compute_ramp (GeglBuffer          *dest1,
              GeglBuffer          *dest2,
              const GeglRectangle *roi,
              gdouble              pct_black,
              gdouble              pct_white,
              gboolean             under_threshold,
              gdouble             *threshold_black,
              gdouble             *threshold_white)
{
  GeglBufferIterator *iter;

  gint    hist1[2000];
  gint    hist2[2000];
  gdouble diff;
  gint    count;

  iter = gegl_buffer_iterator_new (dest1, roi, 0,
                                   babl_format ("Y float"),
                                   GEGL_ACCESS_READ,
                                   GEGL_ABYSS_NONE, 2);
  gegl_buffer_iterator_add (iter, dest2, roi, 0,
                            babl_format ("Y float"),
                            GEGL_ACCESS_READ,
                            GEGL_ABYSS_NONE);

  memset (hist1, 0, sizeof (int) * 2000);
  memset (hist2, 0, sizeof (int) * 2000);
  count = 0;

  while (gegl_buffer_iterator_next (iter))
  {
    gint n_pixels = iter->length;
    gfloat *ptr1  = iter->items[0].data;
    gfloat *ptr2  = iter->items[1].data;

    while (n_pixels--)
      {
        diff = *ptr1 / *ptr2;
        ptr1++;
        ptr2++;

        if (under_threshold)
          {
            if (diff < THRESHOLD && diff >= 0.0f)
              {
                hist2[(int) (diff * 1000)] ++;
                count ++;
              }
          }
        else
          {
            if (diff >= THRESHOLD && diff < 2.0)
              {
                hist1[(int) (diff * 1000)] ++;
                count ++;
              }
          }

      }
  }

  *threshold_black = calculate_threshold (hist1, pct_black, count, 0);
  *threshold_white = calculate_threshold (hist2, pct_white, count, 1);
}

static void
prepare (GeglOperation *operation)
{
  gegl_operation_set_format (operation, "input",
                             babl_format ("Y float"));
  gegl_operation_set_format (operation, "output",
                             babl_format ("Y float"));
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

#include "opencl/gblur-mem-cl.h"
#include "opencl/photocopy.cl.h"

static GeglClRunData *cl_data = NULL;

/* Does what process () does, keeping the blurred images on the GPU */
static gboolean
cl_process (GeglOperation       *operation,
            GeglBuffer          *input,
            GeglBuffer          *output,
            const GeglRectangle *result)
{
  GeglProperties       *o      = GEGL_PROPERTIES (operation);
  const GeglRectangle  *extent = gegl_buffer_get_extent (input);
  const Babl           *ya     = babl_format ("YA float");
  const Babl           *out_format;
  GeglBufferClIterator *i;
  cl_mem                image = NULL, blur1 = NULL, blur2 = NULL;
  cl_mem                hist_mem = NULL;
  cl_int                cl_err = CL_SUCCESS;
  cl_int                stride, x0, y0, width, n_pixels, n_comp;
  cl_float              ramp_down_f, ramp_up_f;
  gboolean              err = FALSE;
  gboolean              ok  = FALSE;
  gint                  hist1[2000];
  gint                  hist2[2000];
  gint                  count, k;
  gdouble               radius, std_dev1, std_dev2;
  gdouble               ramp_down, ramp_up;
  size_t                n, global_ws, local_ws;

  /* The kernels use the sRGB luminance */
  if (babl_format_get_space (gegl_buffer_get_format (input)) !=
      babl_space ("sRGB"))
    return FALSE;

  if (gegl_rectangle_is_empty (result) ||
      ! gegl_rectangle_contains (extent, result))
    return FALSE;

  if (! cl_data)
    {
      const char *kernel_name[] = { "photocopy_hist",
                                    "photocopy_process",
                                    NULL };

      cl_data = gegl_cl_compile_and_build (photocopy_cl_source, kernel_name);
    }

  if (! cl_data)
    return FALSE;

  /* Same as grey_blur_buffer () */
  radius   = MAX (1.0, 10 * (1.0 - o->sharpness));
  radius   = fabs (radius) + 1.0;
  std_dev1 = sqrt (-(radius * radius) / (2 * log (1.0 / 255.0)));

  radius   = fabs (o->mask_radius) + 1.0;
  std_dev2 = sqrt (-(radius * radius) / (2 * log (1.0 / 255.0)));

  n = (size_t) extent->width * extent->height;

  image = gegl_clCreateBuffer (gegl_cl_get_context (), CL_MEM_READ_WRITE,
                               n * 2 * sizeof (cl_float), NULL, &cl_err);
  if (cl_err != CL_SUCCESS)
    goto cleanup;

  blur1 = gegl_clCreateBuffer (gegl_cl_get_context (), CL_MEM_READ_WRITE,
                               n * 2 * sizeof (cl_float), NULL, &cl_err);
  if (cl_err != CL_SUCCESS)
    goto cleanup;

  blur2 = gegl_clCreateBuffer (gegl_cl_get_context (), CL_MEM_READ_WRITE,
                               n * 2 * sizeof (cl_float), NULL, &cl_err);
  if (cl_err != CL_SUCCESS)
    goto cleanup;

  /* gegl:gaussian-blur works on the RaGaBaA float pixels of the input,
   * of which only the luminance is used, so blur (Y * A, A) instead.
   */
  if (! gblur_mem_cl_read (input, extent, ya, image, TRUE))
    goto cleanup;

  if (! gblur_mem_cl_blur (image, blur1, extent->width, extent->height, 2,
                           std_dev1, TRUE) ||
      ! gblur_mem_cl_blur (image, blur2, extent->width, extent->height, 2,
                           std_dev2, TRUE))
    goto cleanup;

  /* compute_ramp (), which only fills hist2 */
  memset (hist1, 0, sizeof (hist1));
  memset (hist2, 0, sizeof (hist2));

  hist_mem = gegl_clCreateBuffer (gegl_cl_get_context (),
                                  CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                  sizeof (hist2), hist2, &cl_err);
  if (cl_err != CL_SUCCESS)
    goto cleanup;

  stride   = extent->width;
  x0       = result->x - extent->x;
  y0       = result->y - extent->y;
  width    = result->width;
  n_pixels = result->width * result->height;

  cl_err = gegl_cl_set_kernel_args (cl_data->kernel[0],
                                    sizeof (cl_mem), &blur1,
                                    sizeof (cl_mem), &blur2,
                                    sizeof (cl_mem), &hist_mem,
                                    sizeof (cl_int), &stride,
                                    sizeof (cl_int), &x0,
                                    sizeof (cl_int), &y0,
                                    sizeof (cl_int), &width,
                                    sizeof (cl_int), &n_pixels,
                                    NULL);
  if (cl_err != CL_SUCCESS)
    goto cleanup;

  local_ws  = 256;
  global_ws = local_ws * CLAMP (n_pixels / 16384, 1, 256);

  cl_err = gegl_clEnqueueNDRangeKernel (gegl_cl_get_command_queue (),
                                        cl_data->kernel[0], 1,
                                        NULL, &global_ws, &local_ws,
                                        0, NULL, NULL);
  if (cl_err != CL_SUCCESS)
    goto cleanup;

  cl_err = gegl_clEnqueueReadBuffer (gegl_cl_get_command_queue (),
                                     hist_mem, CL_TRUE, 0, sizeof (hist2),
                                     hist2, 0, NULL, NULL);
  if (cl_err != CL_SUCCESS)
    goto cleanup;

  /* every pixel counted is in a bin */
  count = 0;
  for (k = 0; k < 2000; k++)
    count += hist2[k];

  ramp_down = calculate_threshold (hist1, o->black, count, 0);
  ramp_up   = calculate_threshold (hist2, o->white, count, 1);

  ramp_down_f = ramp_down;
  ramp_up_f   = ramp_up;

  /* Y float output, or YA float that the iterator can convert on the GPU */
  out_format = gegl_buffer_get_format (output);
  if (out_format != babl_format ("Y float"))
    out_format = ya;
  n_comp = babl_format_get_n_components (out_format);

  i = gegl_buffer_cl_iterator_new (output, result, out_format,
                                   GEGL_CL_BUFFER_WRITE);

  while (gegl_buffer_cl_iterator_next (i, &err) && ! err)
    {
      const GeglRectangle *roi = &i->roi[0];

      x0        = roi->x - extent->x;
      y0        = roi->y - extent->y;
      width     = roi->width;
      global_ws = i->size[0];

      cl_err = gegl_cl_set_kernel_args (cl_data->kernel[1],
                                        sizeof (cl_mem),   &blur1,
                                        sizeof (cl_mem),   &blur2,
                                        sizeof (cl_mem),   &i->tex[0],
                                        sizeof (cl_int),   &stride,
                                        sizeof (cl_int),   &x0,
                                        sizeof (cl_int),   &y0,
                                        sizeof (cl_int),   &width,
                                        sizeof (cl_int),   &n_comp,
                                        sizeof (cl_float), &ramp_down_f,
                                        sizeof (cl_float), &ramp_up_f,
                                        NULL);
      if (cl_err == CL_SUCCESS)
        cl_err = gegl_clEnqueueNDRangeKernel (gegl_cl_get_command_queue (),
                                              cl_data->kernel[1], 1,
                                              NULL, &global_ws, NULL,
                                              0, NULL, NULL);
      if (cl_err != CL_SUCCESS)
        {
          gegl_buffer_cl_iterator_stop (i);
          goto cleanup;
        }
    }

  ok = ! err;

cleanup:
  if (hist_mem)
    gegl_clReleaseMemObject (hist_mem);
  if (blur2)
    gegl_clReleaseMemObject (blur2);
  if (blur1)
    gegl_clReleaseMemObject (blur1);
  if (image)
    gegl_clReleaseMemObject (image);

  return ok;
}

static gboolean
process (GeglOperation       *operation,
         GeglBuffer          *input,
         GeglBuffer          *output,
         const GeglRectangle *result,
         gint                 level)
{
  GeglProperties *o = GEGL_PROPERTIES (operation);

  GeglBufferIterator *iter;

  GeglBuffer *dest1;
  GeglBuffer *dest2;

  gdouble diff;
  gdouble ramp_down;
  gdouble ramp_up;
  gdouble mult;

  if (level == 0 && gegl_operation_use_opencl (operation))
    if (cl_process (operation, input, output, result))
      return TRUE;

  grey_blur_buffer (input, o->sharpness, o->mask_radius, &dest1, &dest2);

  compute_ramp (dest1, dest2, result,
                o->black, o->white, TRUE,
                &ramp_down, &ramp_up);

  iter = gegl_buffer_iterator_new (dest1, result, 0,
                                   babl_format ("Y float"),
                                   GEGL_ACCESS_READ,
                                   GEGL_ABYSS_NONE, 4);
  gegl_buffer_iterator_add (iter, dest2, result, 0,
                            babl_format ("Y float"),
                            GEGL_ACCESS_READ,
                            GEGL_ABYSS_NONE);
  gegl_buffer_iterator_add (iter, output, result, 0,
                            babl_format ("Y float"),
                            GEGL_ACCESS_WRITE,
                            GEGL_ABYSS_NONE);

  while (gegl_buffer_iterator_next (iter))
    {
      gint    n_pixels  = iter->length;
      gfloat *ptr1      = iter->items[0].data;
      gfloat *ptr2      = iter->items[1].data;
      gfloat *out_pixel = iter->items[2].data;

      while (n_pixels--)
        {
          diff = *ptr1 / *ptr2;
          if (diff < THRESHOLD)
            {
              if (ramp_down == 0.0)
                *out_pixel = 0.0;
              else
                {
                  mult = (ramp_down - MIN (ramp_down,
                                           (THRESHOLD - diff))) / ramp_down;
                  *out_pixel = *ptr1 * mult;
                }
            }
          else
            {
              if (ramp_up == 0.0)
                mult = 1.0;
              else
                mult = MIN (ramp_up,
                            (diff - THRESHOLD)) / ramp_up;

              *out_pixel = mult + *ptr1 - mult * *ptr1;
            }

          ptr1++;
          ptr2++;
          out_pixel++;
        }
    }

  g_object_unref (dest1);
  g_object_unref (dest2);

  return TRUE;
}

static void
gegl_op_class_init (GeglOpClass *klass)
{
  GeglOperationClass       *operation_class;
  GeglOperationFilterClass *filter_class;

  operation_class = GEGL_OPERATION_CLASS (klass);
  filter_class    = GEGL_OPERATION_FILTER_CLASS (klass);

  operation_class->prepare                 = prepare;
  operation_class->get_required_for_output = get_required_for_output;
  operation_class->get_cached_region       = get_cached_region;
  operation_class->threaded                = FALSE;
  filter_class->process                    = process;
  operation_class->opencl_support          = TRUE;

  gegl_operation_class_set_keys (operation_class,
    "name",          "gegl:photocopy",
    "categories",    "artistic",
    "license",       "GPL3+",
    "title",         _("Photocopy"),
    "reference-hash", "711b12475b7bdab15f9f29bdbf4379b5",
    "description", _("Simulate color distortion produced by a copy machine"),
    NULL);
}
#endif
