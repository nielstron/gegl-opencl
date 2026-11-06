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
 *
 * Copyright 2012 Maxime Nicco <maxime.nicco@gmail.com>
 */

#include "config.h"
#include <glib/gi18n-lib.h>

#ifdef GEGL_PROPERTIES

property_int   (holdness, _("Dulling"), 2)
   value_range (1, 8)
   description (_("A high value lowers the randomness of the noise"))

property_double (lightness_distance, _("Lightness"), 40.0)
   value_range  (0.0, 100.0)

property_double (chroma_distance, _("Chroma"), 40.0)
   value_range  (0.0, 100.0)

property_double (hue_distance, _("Hue"), 3.0)
   value_range  (0.0, 180.0)

property_seed   (seed, _("Random seed"), rand)

#else

#define GEGL_OP_POINT_FILTER
#define GEGL_OP_NAME     noise_cie_lch
#define GEGL_OP_C_SOURCE noise-cie-lch.c

#include "gegl-op.h"
#include "gegl.h"
#include <stdio.h>
#include <stdlib.h>

static gfloat
randomize_value (gfloat      now,
                 gfloat      min,
                 gfloat      max,
                 gboolean    wraps_around,
                 gfloat      rand_max,
                 gint        holdness,
                 gint        x,
                 gint        y,
                 gint        n,
                 GeglRandom *rand)
{
  gint    flag, i;
  gfloat rand_val, new_val, steps;

  steps = max - min + 0.5;
  rand_val = gegl_random_float (rand, x, y, 0, n++);

  for (i = 1; i < holdness; i++)
  {
    float tmp = gegl_random_float (rand, x, y, 0, n++);
    if (tmp < rand_val)
      rand_val = tmp;
  }

  flag = (gegl_random_float (rand, x, y, 0, n) < 0.5) ? -1 : 1;
  new_val = now + flag * fmod (rand_max * rand_val, steps);

  if (new_val < min)
  {
    if (wraps_around)
      new_val += steps;
    else
      new_val = min;
  }

  if (max < new_val)
  {
    if (wraps_around)
      new_val -= steps;
    else
      new_val = max;
  }

  return new_val;
}

static void
prepare (GeglOperation *operation)
{
  const Babl *space  = gegl_operation_get_source_space (operation, "input");
  const Babl *source = gegl_operation_get_source_format (operation, "input");
  const Babl *lch    = babl_format_with_space ("CIE LCH(ab) alpha float",
                                               space);
  const Babl *rgbaf  = babl_format_with_space ("RGBA float", space);

  /* There is no OpenCL color conversion for LCH, so for the most common
   * input formats the OpenCL kernel converts from and to RGB itself, with
   * the same math as babl.  8-bit input is passed through untouched, so
   * that the kernel can linearize it with the exact values babl uses.
   */
  if (gegl_operation_use_opencl (operation) &&
      (source == rgbaf ||
       source == babl_format_with_space ("R'G'B'A u8", space) ||
       source == babl_format_with_space ("R'G'B' u8", space)))
    {
      gegl_operation_set_format (operation, "input",  source);
      gegl_operation_set_format (operation, "output", rgbaf);
    }
  else
    {
      gegl_operation_set_format (operation, "input",  lch);
      gegl_operation_set_format (operation, "output", lch);
    }
}

static gboolean
process_lch (GeglOperation       *operation,
             void                *in_buf,
             void                *out_buf,
             glong                n_pixels,
             const GeglRectangle *roi,
             gint                 level)
{
  GeglProperties *o = GEGL_PROPERTIES (operation);
  gfloat         *GEGL_ALIGNED in_pixel;
  gfloat         *GEGL_ALIGNED out_pixel;
  GeglRectangle   whole_region;
  gfloat          lightness, chroma, hue, alpha;
  gint            i;
  gint            x, y;

  in_pixel  = in_buf;
  out_pixel = out_buf;

  x = roi->x;
  y = roi->y;

  whole_region = *(gegl_operation_source_get_bounding_box (operation, "input"));

  for (i = 0; i < n_pixels; i++)
  {
    /* n is independent from the roi, but from the whole image */
    gint n = (3 * o->holdness + 4) * (x + whole_region.width * y);

    lightness = in_pixel[0];
    chroma    = in_pixel[1];
    hue       = in_pixel[2];
    alpha     = in_pixel[3];

    if ((o->hue_distance > 0) && (chroma > 0))
      hue = randomize_value (hue, 0.0, 359.0, TRUE, o->hue_distance,
                             o->holdness, x, y, n, o->rand);

    n += o->holdness + 1;
    if (o->chroma_distance > 0) {
      if (chroma == 0)
        hue = gegl_random_float_range (o->rand, x, y, 0, n, 0.0, 360.0);
      chroma = randomize_value (chroma, 0.0, 100.0, FALSE, o->chroma_distance,
                                o->holdness, x, y, n+1, o->rand);
    }

    n += o->holdness + 2;
    if (o->lightness_distance > 0)
      lightness = randomize_value (lightness, 0.0, 100.0, FALSE,
                                   o->lightness_distance, o->holdness,
                                   x, y, n, o->rand);

    /*n += o->holdness + 1*/
    out_pixel[0] = lightness;
    out_pixel[1] = chroma;
    out_pixel[2] = hue;
    out_pixel[3] = alpha;

    in_pixel  += 4;
    out_pixel += 4;

    x++;
    if (x >= roi->x + roi->width)
      {
        x = roi->x;
        y++;
      }
  }

  return TRUE;
}

/* also the CPU fallback when the OpenCL formats were chosen in prepare () */
static gboolean
process (GeglOperation       *operation,
         void                *in_buf,
         void                *out_buf,
         glong                n_pixels,
         const GeglRectangle *roi,
         gint                 level)
{
  const Babl *in_format  = gegl_operation_get_format (operation, "input");
  const Babl *out_format = gegl_operation_get_format (operation, "output");
  const Babl *lch;
  gfloat     *buf;

  lch = babl_format_with_space ("CIE LCH(ab) alpha float", out_format);

  if (in_format == lch && out_format == lch)
    return process_lch (operation, in_buf, out_buf, n_pixels, roi, level);

  buf = gegl_malloc (n_pixels * 4 * sizeof (gfloat));

  babl_process (babl_fish (in_format, lch), in_buf, buf, n_pixels);
  process_lch (operation, buf, buf, n_pixels, roi, level);
  babl_process (babl_fish (lch, out_format), buf, out_buf, n_pixels);

  gegl_free (buf);

  return TRUE;
}

#include "opencl/gegl-cl.h"
#include "opencl/noise-cie-lch.cl.h"

static GeglClRunData *cl_data = NULL;

/* The matrices babl's "RGBA float" <-> "CIE Lab alpha float" conversions
 * use for the space of format.  The RGB to XYZ matrix is read back from
 * babl by converting the three primaries.
 */
static void
get_lab_matrices (const Babl *format,
                  cl_float   *rgb_to_lab,
                  cl_float   *lab_to_rgb)
{
  const gfloat white[3] = { 0.964202880f, 1.000000000f, 0.824905400f };
  const gfloat rgba[12] = { 1.0f, 0.0f, 0.0f, 1.0f,
                            0.0f, 1.0f, 0.0f, 1.0f,
                            0.0f, 0.0f, 1.0f, 1.0f };
  gfloat       xyza[12];
  gdouble      m[9];
  gdouble      inv[9];
  gdouble      det;
  gint         i;

  babl_process (babl_fish (format,
                           babl_format_with_space ("CIE XYZ alpha float",
                                                   format)),
                rgba, xyza, 3);

  /* column c of the matrix is the XYZ of primary c */
  for (i = 0; i < 9; i++)
    m[i] = xyza[(i % 3) * 4 + i / 3];

  det = m[0] * (m[4] * m[8] - m[7] * m[5]) -
        m[1] * (m[3] * m[8] - m[5] * m[6]) +
        m[2] * (m[3] * m[7] - m[4] * m[6]);

  inv[0] = (m[4] * m[8] - m[7] * m[5]) / det;
  inv[1] = (m[2] * m[7] - m[1] * m[8]) / det;
  inv[2] = (m[1] * m[5] - m[2] * m[4]) / det;
  inv[3] = (m[5] * m[6] - m[3] * m[8]) / det;
  inv[4] = (m[0] * m[8] - m[2] * m[6]) / det;
  inv[5] = (m[3] * m[2] - m[0] * m[5]) / det;
  inv[6] = (m[3] * m[7] - m[6] * m[4]) / det;
  inv[7] = (m[6] * m[1] - m[0] * m[7]) / det;
  inv[8] = (m[0] * m[4] - m[3] * m[1]) / det;

  for (i = 0; i < 9; i++)
    {
      rgb_to_lab[i] = (gfloat) m[i] / white[i / 3];
      lab_to_rgb[i] = (gfloat) inv[i] * white[i % 3];
    }

  for (i = 9; i < 16; i++)
    rgb_to_lab[i] = lab_to_rgb[i] = 0.0f;
}

static gboolean
cl_process (GeglOperation       *operation,
            cl_mem               in,
            cl_mem               out,
            size_t               global_worksize,
            const GeglRectangle *roi,
            gint                 level)
{
  GeglProperties *o  = GEGL_PROPERTIES (operation);
  GeglRectangle  *wr = gegl_operation_source_get_bounding_box (operation,
                                                               "input");
  cl_int          cl_err             = 0;
  cl_mem          cl_random_data;
  cl_ushort4      rand;
  cl_int          wr_width           = wr->width;
  cl_int          holdness           = o->holdness;
  cl_float        lightness_distance = o->lightness_distance;
  cl_float        chroma_distance    = o->chroma_distance;
  cl_float        hue_distance       = o->hue_distance;
  cl_float16      rgb_to_lab;
  cl_float16      lab_to_rgb;
  const Babl     *in_format  = gegl_operation_get_format (operation, "input");
  const Babl     *out_format = gegl_operation_get_format (operation, "output");
  cl_kernel       kernel;
  cl_mem          cl_lut      = NULL;
  cl_mem          cl_gray_lch = NULL;

  if (!cl_data)
    {
      const char *kernel_name[] = { "cl_noise_cie_lch_lch",
                                    "cl_noise_cie_lch",
                                    "cl_noise_cie_lch_u8",
                                    NULL };
      cl_data = gegl_cl_compile_and_build (noise_cie_lch_cl_source,
                                           kernel_name);
    }

  if (!cl_data)
    return TRUE;

  gegl_cl_random_get_ushort4 (o->rand, &rand);

  cl_random_data = gegl_cl_load_random_data (&cl_err);
  CL_CHECK;

  get_lab_matrices (out_format, rgb_to_lab.s, lab_to_rgb.s);

  if (out_format != babl_format_with_space ("RGBA float", out_format))
    {
      kernel = cl_data->kernel[0];
    }
  else if (in_format == out_format)
    {
      kernel = cl_data->kernel[1];
    }
  else
    {
      const Babl *lch = babl_format_with_space ("CIE LCH(ab) alpha float",
                                                out_format);
      cl_int      bpp = babl_format_get_bytes_per_pixel (in_format);
      guint8      gray[256 * 4];
      gfloat      tmp[256 * 4];
      gfloat      lut[256];
      gint        i;

      kernel = cl_data->kernel[2];

      for (i = 0; i < 256 * bpp; i++)
        gray[i] = (i % bpp) < 3 ? i / bpp : 255;

      babl_process (babl_fish (in_format, out_format), gray, tmp, 256);

      for (i = 0; i < 256; i++)
        lut[i] = tmp[i * 4];

      cl_lut = gegl_clCreateBuffer (gegl_cl_get_context (),
                                    CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY,
                                    sizeof (lut), lut, &cl_err);
      CL_CHECK;

      babl_process (babl_fish (in_format, lch), gray, tmp, 256);

      cl_gray_lch = gegl_clCreateBuffer (gegl_cl_get_context (),
                                         CL_MEM_COPY_HOST_PTR |
                                         CL_MEM_READ_ONLY,
                                         sizeof (tmp), tmp, &cl_err);
      CL_CHECK;

      cl_err = gegl_clSetKernelArg (kernel, 14, sizeof (cl_mem), &cl_lut);
      CL_CHECK;
      cl_err = gegl_clSetKernelArg (kernel, 15, sizeof (cl_mem),
                                    &cl_gray_lch);
      CL_CHECK;
      cl_err = gegl_clSetKernelArg (kernel, 16, sizeof (cl_int), &bpp);
      CL_CHECK;
    }

  cl_err = gegl_cl_set_kernel_args (kernel,
                                    sizeof (cl_mem),     &in,
                                    sizeof (cl_mem),     &out,
                                    sizeof (cl_mem),     &cl_random_data,
                                    sizeof (cl_ushort4), &rand,
                                    sizeof (cl_int),     &roi->x,
                                    sizeof (cl_int),     &roi->y,
                                    sizeof (cl_int),     &roi->width,
                                    sizeof (cl_int),     &wr_width,
                                    sizeof (cl_int),     &holdness,
                                    sizeof (cl_float),   &lightness_distance,
                                    sizeof (cl_float),   &chroma_distance,
                                    sizeof (cl_float),   &hue_distance,
                                    sizeof (cl_float16), &rgb_to_lab,
                                    sizeof (cl_float16), &lab_to_rgb,
                                    NULL);
  CL_CHECK;

  cl_err = gegl_clEnqueueNDRangeKernel (gegl_cl_get_command_queue (),
                                        kernel, 1,
                                        NULL, &global_worksize, NULL,
                                        0, NULL, NULL);
  CL_CHECK;

  if (cl_lut)
    gegl_clReleaseMemObject (cl_lut);
  if (cl_gray_lch)
    gegl_clReleaseMemObject (cl_gray_lch);

  return FALSE;

error:
  if (cl_lut)
    gegl_clReleaseMemObject (cl_lut);
  if (cl_gray_lch)
    gegl_clReleaseMemObject (cl_gray_lch);

  return TRUE;
}

static void
gegl_op_class_init (GeglOpClass *klass)
{
  GeglOperationClass            *operation_class;
  GeglOperationPointFilterClass *point_filter_class;

  operation_class    = GEGL_OPERATION_CLASS (klass);
  point_filter_class = GEGL_OPERATION_POINT_FILTER_CLASS (klass);

  operation_class->prepare        = prepare;
  operation_class->opencl_support = TRUE;
  point_filter_class->process     = process;
  point_filter_class->cl_process  = cl_process;

  gegl_operation_class_set_keys (operation_class,
    "name",               "gegl:noise-cie-lch",
    "title",              _("Add CIE Lch Noise"),
    "categories",         "noise",
    "position-dependent", "true",
    "reference-hash",     "cf75f341258e4eaedd9586f8eedb67c1",
    "description", _("Randomize lightness, chroma and hue independently"),
    NULL);
}

#endif
