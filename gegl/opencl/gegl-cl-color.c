/* This file is part of GEGL
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
 * Copyright 2012 Victor Oliveira (victormatheus@gmail.com)
 */


/* Here we have implemented in OpenCL a subset of color space conversions provided by BABL
   that are commonly used. For now there is no support for conversions that need a intermediate
   representation (ex: A->B, B->C, C->D), just among two color spaces.
*/

#include "config.h"

#include <math.h>

#include "gegl.h"
#include "gegl/gegl-debug.h"
#include "gegl-cl.h"
#include "gegl-cl-color.h"

#include "operations/opencl/colors.cl.h"
#include "operations/opencl/colors-8bit-lut.cl.h"

static GHashTable  *color_kernels_hash = NULL;

typedef struct
{
  const Babl *from_fmt;
  const Babl *to_fmt;
  const char *kernel_name;
  cl_kernel   kernel;
  GArray     *variants;
} ColorConversionInfo;

static guint
color_kernels_hash_hashfunc (gconstpointer key)
{
  const ColorConversionInfo *ekey = key;
  return GPOINTER_TO_INT (ekey->from_fmt) ^ GPOINTER_TO_INT (ekey->to_fmt);
}

static gboolean
color_kernels_hash_equalfunc (gconstpointer a,
                              gconstpointer b)
{
  const ColorConversionInfo *ea = a;
  const ColorConversionInfo *eb = b;

  if (ea->from_fmt == eb->from_fmt &&
      ea->to_fmt   == eb->to_fmt)
    {
      return TRUE;
    }
  return FALSE;
}

static gboolean
gegl_cl_color_load_conversion_set (ColorConversionInfo  *conversions,
                                   gint                  num_conversions,
                                   GeglClRunData       **kernels,
                                   const gchar          *source)
{
  const char **kernel_names = g_newa (const char *, num_conversions + 1);

  for (int i = 0; i < num_conversions; ++i)
    {
      kernel_names[i] = conversions[i].kernel_name;
    }

  kernel_names[num_conversions] = NULL;

  *kernels = gegl_cl_compile_and_build (source, kernel_names);

  if (!(*kernels))
    {
      return FALSE;
    }

  for (int i = 0; i < num_conversions; ++i)
    {
      ColorConversionInfo *info;

      conversions[i].kernel = (*kernels)->kernel[i];

      /* A conversion listed more than once has variants, see
       * pick_variant ().
       */
      info = g_hash_table_lookup (color_kernels_hash, &conversions[i]);
      if (info)
        {
          if (! info->variants)
            {
              info->variants = g_array_new (FALSE, FALSE, sizeof (cl_kernel));
              g_array_append_val (info->variants, info->kernel);
            }
          g_array_append_val (info->variants, conversions[i].kernel);
          continue;
        }

      info = g_new (ColorConversionInfo, 1);
      *info = conversions[i];
      info->variants = NULL;

      g_hash_table_insert (color_kernels_hash, info, info);
    }

  return TRUE;
}

static gboolean
run_color_kernel (cl_kernel    kernel,
                  cl_mem       in_tex,
                  cl_mem       out_tex,
                  const size_t size)
{
  cl_int cl_err = 0;

  cl_err = gegl_cl_set_kernel_args (kernel,
                                    sizeof(cl_mem), &in_tex,
                                    sizeof(cl_mem), &out_tex,
                                    NULL);
  CL_CHECK;

  cl_err = gegl_clEnqueueNDRangeKernel (gegl_cl_get_command_queue (),
                                        kernel, 1,
                                        NULL, &size, NULL,
                                        0, NULL, NULL);
  CL_CHECK;
  return FALSE;

error:
  return TRUE;
}

#define PROBE_PIXELS 4096

static guint32
probe_hash (guint32 x)
{
  x ^= x >> 16;
  x *= 0x7feb352d;
  x ^= x >> 15;
  x *= 0x846ca68b;
  x ^= x >> 16;
  return x;
}

/* Fills @buf with PROBE_PIXELS pixels of @format that tell the variants of
 * a conversion to @out_format apart.
 */
static void
make_probe_pixels (const Babl *format,
                   const Babl *out_format,
                   guint8     *buf)
{
  gint     n_components = babl_format_get_n_components (format);
  gboolean has_alpha    = babl_format_has_alpha (format);
  gint     n_out        = babl_format_get_n_components (out_format);
  gint     i, c;

  if (babl_format_get_bytes_per_pixel (format) == n_components)
    {
      for (i = 0; i < PROBE_PIXELS * n_components; i++)
        buf[i] = probe_hash (i);
      return;
    }

  if (babl_format_get_bytes_per_pixel (out_format) == n_out)
    {
      /* Variants that differ in the last bit only give a different 8-bit
       * result next to rounding boundaries, so convert values near them
       * back to the input format.
       */
      gchar      *name = g_strdup_printf ("%s float",
                                          babl_get_name (babl_format_get_model (out_format)));
      const Babl *out_float = babl_format (name);
      gfloat     *tmp  = g_new (gfloat, PROBE_PIXELS * n_out);
      gfloat     *in   = (gfloat *) buf;

      g_free (name);

      for (i = 0; i < PROBE_PIXELS * n_out; i++)
        tmp[i] = ((probe_hash (i) & 0xff) + 0.5f) / 255.0f;

      babl_process (babl_fish (out_float, format), tmp, in, PROBE_PIXELS);

      for (i = 0; i < PROBE_PIXELS; i++)
        for (c = 0; c < n_components; c++)
          {
            guint32 h = probe_hash (i * n_components + c + PROBE_PIXELS * 4);
            gfloat *v = &in[i * n_components + c];

            if (has_alpha && c == n_components - 1)
              *v = (i & 3) ? (h >> 8) / 16777216.0f : 1.0f;
            else
              for (gint steps = h & 3; steps; steps--)
                *v = nextafterf (*v, (h & 4) ? G_MAXFLOAT : -G_MAXFLOAT);
          }

      g_free (tmp);
      return;
    }

  for (i = 0; i < PROBE_PIXELS; i++)
    for (c = 0; c < n_components; c++)
      {
        guint32 h = probe_hash (i * n_components + c);
        gfloat *v = &((gfloat *) buf)[i * n_components + c];

        if (has_alpha && c == n_components - 1)
          *v = (i & 3) ? (h >> 8) / 16777216.0f : 1.0f;
        else
          *v = -0.1f + 1.3f * ((h >> 8) / 16777216.0f);
      }
}

/* babl picks between conversion paths at runtime, by measuring their speed,
 * and the paths can differ in the last bit.  Run each kernel variant on
 * probe pixels and keep the one closest to what babl computes.
 */
static void
pick_variant (gpointer key,
              gpointer value,
              gpointer user_data)
{
  ColorConversionInfo *info    = value;
  gint                 in_bpp  = babl_format_get_bytes_per_pixel (info->from_fmt);
  gint                 out_bpp = babl_format_get_bytes_per_pixel (info->to_fmt);
  guint8              *in;
  guint8              *ref;
  guint8              *out;
  cl_mem               in_tex  = NULL;
  cl_mem               out_tex = NULL;
  gint                 best_mismatches = G_MAXINT;
  cl_int               cl_err = 0;

  if (! info->variants)
    return;

  in  = g_malloc (PROBE_PIXELS * in_bpp);
  ref = g_malloc (PROBE_PIXELS * out_bpp);
  out = g_malloc (PROBE_PIXELS * out_bpp);

  make_probe_pixels (info->from_fmt, info->to_fmt, in);
  babl_process (babl_fish (info->from_fmt, info->to_fmt), in, ref, PROBE_PIXELS);

  in_tex = gegl_clCreateBuffer (gegl_cl_get_context (),
                                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                PROBE_PIXELS * in_bpp, in, &cl_err);
  CL_CHECK;
  out_tex = gegl_clCreateBuffer (gegl_cl_get_context (), CL_MEM_WRITE_ONLY,
                                 PROBE_PIXELS * out_bpp, NULL, &cl_err);
  CL_CHECK;

  for (guint v = 0; v < info->variants->len; v++)
    {
      cl_kernel kernel     = g_array_index (info->variants, cl_kernel, v);
      gint      mismatches = 0;

      if (run_color_kernel (kernel, in_tex, out_tex, PROBE_PIXELS))
        goto error;

      cl_err = gegl_clEnqueueReadBuffer (gegl_cl_get_command_queue (),
                                         out_tex, CL_TRUE, 0,
                                         PROBE_PIXELS * out_bpp, out,
                                         0, NULL, NULL);
      CL_CHECK;

      for (gint i = 0; i < PROBE_PIXELS * out_bpp; i++)
        mismatches += (out[i] != ref[i]);

      GEGL_NOTE (GEGL_DEBUG_OPENCL, "%s -> %s variant %u: %d mismatching bytes",
                 babl_get_name (info->from_fmt), babl_get_name (info->to_fmt),
                 v, mismatches);

      if (mismatches < best_mismatches)
        {
          best_mismatches = mismatches;
          info->kernel    = kernel;
        }
    }

error:
  if (in_tex)
    gegl_clReleaseMemObject (in_tex);
  if (out_tex)
    gegl_clReleaseMemObject (out_tex);

  g_free (in);
  g_free (ref);
  g_free (out);

  g_array_free (info->variants, TRUE);
  info->variants = NULL;
}

/* Converts each 8-bit value from @from to @to with babl, alpha opaque, and
 * stores the first output component of each in @table.
 */
static void
babl_u8_table (const gchar *from,
               const gchar *to,
               gfloat      *table)
{
  const Babl *from_fmt = babl_format (from);
  const Babl *to_fmt   = babl_format (to);
  gint        from_n   = babl_format_get_n_components (from_fmt);
  gint        to_n     = babl_format_get_n_components (to_fmt);
  guint8      src[256 * 4];
  gfloat      dst[256 * 4];

  g_return_if_fail (babl_format_get_bytes_per_pixel (from_fmt) == from_n);
  g_return_if_fail (babl_format_get_bytes_per_pixel (to_fmt) ==
                    to_n * sizeof (gfloat));

  for (gint i = 0; i < 256; i++)
    for (gint c = 0; c < from_n; c++)
      src[i * from_n + c] = (c == 3) ? 255 : i;

  babl_process (babl_fish (from_fmt, to_fmt), src, dst, 256);

  for (gint i = 0; i < 256; i++)
    table[i] = dst[i * to_n];
}

static void
append_float (GString *source,
              gfloat   value)
{
  gchar buf[G_ASCII_DTOSTR_BUF_SIZE];

  g_string_append (source, g_ascii_formatd (buf, sizeof (buf), "%.9e", value));
  g_string_append_c (source, 'f');
}

static void
append_table (GString      *source,
              const gchar  *name,
              const gfloat *table)
{
  g_string_append_printf (source, "__constant float %s[256] = {\n", name);
  for (gint i = 0; i < 256; i++)
    {
      g_string_append (source, "  ");
      append_float (source, table[i]);
      g_string_append (source, ",\n");
    }
  g_string_append (source, "};\n\n");
}

static void
append_u8_table (GString     *source,
                 const gchar *name,
                 const gchar *from,
                 const gchar *to)
{
  gfloat table[256];

  babl_u8_table (from, to, table);
  append_table (source, name, table);
}

/* The two 8-bit sRGB to linear tables babl has: computed in single
 * precision (extensions/gimp-8bit.c) and in double precision
 * (extensions/gegl-fixups.c).
 */
static void
append_gamma_u8_tables (GString *source)
{
  gfloat u8[256];
  gfloat table[256];

  babl_u8_table ("Y u8", "Y float", u8);
  babl_process (babl_fish ("Y' float", "Y float"), u8, table, 256);
  append_table (source, "gamma_u8_lut_f", table);

  for (gint i = 0; i < 256; i++)
    {
      gfloat direct = i / 255.0;
      double value  = direct;

      if (value > 0.04045)
        table[i] = pow ((value + 0.055) / 1.055, 2.4);
      else
        table[i] = value / 12.92;
    }
  append_table (source, "gamma_u8_lut_d", table);
}

/* The kernels reproduce babl's conversions bit-exactly, where babl's
 * constants and lookup tables are defined at runtime: the luminance weights
 * come from the space, and babl may pick a different 8-bit table depending
 * on which conversion path it measured as fastest.  Generate them here
 * instead of hard-coding them in the kernel source.
 */
static GString *
babl_kernel_header (void)
{
  GString             *header    = g_string_new (NULL);
  cl_device_fp_config  fp_config = 0;
  double               lum[3];

  babl_space_get_rgb_luminance (babl_format_get_space (babl_format ("RGBA float")),
                                &lum[0], &lum[1], &lum[2]);

  g_string_append (header, "#define BABL_LUMINANCE_RED   ");
  append_float (header, lum[0]);
  g_string_append (header, "\n#define BABL_LUMINANCE_GREEN ");
  append_float (header, lum[1]);
  g_string_append (header, "\n#define BABL_LUMINANCE_BLUE  ");
  append_float (header, lum[2]);
  g_string_append (header, "\n\n");

  append_u8_table (header, "babl_u8_to_float", "Y u8", "Y float");
  append_u8_table (header, "rgbau8_to_rgba_gamma_f_lut",
                   "RGBA u8", "R'G'B'A float");

  /* gegl_cl_compile_and_build () passes -cl-fp32-correctly-rounded-divide-sqrt
   * when the device supports it.
   */
  if (gegl_clGetDeviceInfo (gegl_cl_get_device (), CL_DEVICE_SINGLE_FP_CONFIG,
                            sizeof (fp_config), &fp_config, NULL) == CL_SUCCESS &&
      (fp_config & CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT))
    g_string_append (header, "#define BABL_CORRECTLY_ROUNDED_SQRT\n\n");

  return header;
}

gboolean
gegl_cl_color_compile_kernels (void)
{
  static GeglClRunData *float_kernels = NULL;
  static GeglClRunData *lut8_kernels = NULL;
  gboolean result = TRUE;
  GString *float_source;
  GString *lut8_source;

  ColorConversionInfo float_conversions[] = {
    { babl_format ("RGBA u8"), babl_format ("RGBA float"), "rgbau8_to_rgbaf", NULL },
    { babl_format ("RGBA float"), babl_format ("RGBA u8"), "rgbaf_to_rgbau8", NULL },

    { babl_format ("RGBA float"), babl_format("RaGaBaA float"), "rgbaf_to_ragabaf", NULL },
    { babl_format ("RaGaBaA float"), babl_format("RGBA float"), "ragabaf_to_rgbaf", NULL },
    { babl_format ("RGBA u8"), babl_format("RaGaBaA float"), "rgbau8_to_ragabaf", NULL },
    { babl_format ("RaGaBaA float"), babl_format("RGBA u8"), "ragabaf_to_rgbau8", NULL },

    { babl_format ("RGBA float"), babl_format("R'G'B'A float"), "rgbaf_to_rgba_gamma_f", NULL },
    { babl_format ("R'G'B'A float"), babl_format("RGBA float"), "rgba_gamma_f_to_rgbaf", NULL },
    { babl_format ("RGBA u8"), babl_format("R'G'B'A float"), "rgbau8_to_rgba_gamma_f", NULL },
    { babl_format ("R'G'B'A float"), babl_format("RGBA u8"), "rgba_gamma_f_to_rgbau8", NULL },
    { babl_format ("R'G'B'A float"), babl_format("RGBA u8"), "rgba_gamma_f_to_rgbau8_assoc", NULL },

    { babl_format ("RGBA float"), babl_format("Y'CbCrA float"), "rgbaf_to_ycbcraf", NULL },
    { babl_format ("Y'CbCrA float"), babl_format("RGBA float"), "ycbcraf_to_rgbaf", NULL },
    { babl_format ("RGBA u8"), babl_format("Y'CbCrA float"), "rgbau8_to_ycbcraf", NULL },
    { babl_format ("Y'CbCrA float"), babl_format("RGBA u8"), "ycbcraf_to_rgbau8", NULL },

    { babl_format ("RGB u8"), babl_format("RGBA float"), "rgbu8_to_rgbaf", NULL },
    { babl_format ("RGBA float"), babl_format("RGB u8"), "rgbaf_to_rgbu8", NULL },

    { babl_format ("Y u8"), babl_format("Y float"), "yu8_to_yf", NULL },

    { babl_format ("RGBA float"), babl_format("YA float"), "rgbaf_to_yaf", NULL },
    { babl_format ("RGBA float"), babl_format("YA float"), "rgbaf_to_yaf_grey", NULL },
    { babl_format ("YA float"), babl_format("RGBA float"), "yaf_to_rgbaf", NULL },
    { babl_format ("RGBA u8"), babl_format("YA float"), "rgbau8_to_yaf", NULL },
    { babl_format ("RGBA u8"), babl_format("YA float"), "rgbau8_to_yaf_grey", NULL },
    { babl_format ("YA float"), babl_format("RGBA u8"), "yaf_to_rgbau8", NULL },
    { babl_format ("RaGaBaA float"), babl_format("YA float"), "ragabaf_to_yaf", NULL },
    { babl_format ("RaGaBaA float"), babl_format("YA float"), "ragabaf_to_yaf_grey", NULL },

    { babl_format ("RGBA float"), babl_format("R'G'B'A u8"), "rgbaf_to_rgba_gamma_u8", NULL },
    { babl_format ("RGBA float"), babl_format("R'G'B'A u8"), "rgbaf_to_rgba_gamma_u8_assoc", NULL },
    { babl_format ("RGBA float"), babl_format("R'G'B' u8"), "rgbaf_to_rgb_gamma_u8", NULL },

    { babl_format ("RaGaBaA float"), babl_format("R'G'B'A u8"), "ragabaf_to_rgba_gamma_u8", NULL },
    { babl_format ("RaGaBaA float"), babl_format("R'G'B' u8"), "ragabaf_to_rgb_gamma_u8", NULL },

    { babl_format ("YA float"), babl_format("R'G'B'A u8"), "yaf_to_rgba_gamma_u8", NULL },
    { babl_format ("YA float"), babl_format("R'G'B' u8"), "yaf_to_rgb_gamma_u8", NULL },

    { babl_format ("YA float"), babl_format ("RaGaBaA float"), "yaf_to_ragabaf", NULL },

    { babl_format ("Y float"), babl_format ("RaGaBaA float"), "yf_to_ragabaf", NULL },

    { babl_format ("RGBA float"), babl_format ("RGB float"), "rgbaf_to_rgbf", NULL },

    { babl_format ("R'G'B' float"), babl_format ("RGBA float"), "rgb_gamma_f_to_rgbaf", NULL },

    /* Reuse some conversions */
    { babl_format ("R'G'B' u8"), babl_format ("R'G'B'A float"), "rgbu8_to_rgbaf", NULL },
    { babl_format ("R'G'B'A u8"), babl_format ("R'G'B'A float"), "rgbau8_to_rgbaf", NULL },

    { babl_format ("R'G'B' float"), babl_format ("RaGaBaA float"), "rgb_gamma_f_to_rgbaf", NULL },
  };

  ColorConversionInfo lut8_conversions[] = {
    { babl_format ("R'G'B'A u8"), babl_format("RGBA float"), "rgba_gamma_u8_to_rgbaf", NULL },
    { babl_format ("R'G'B'A u8"), babl_format("RaGaBaA float"), "rgba_gamma_u8_to_ragabaf", NULL },
    { babl_format ("R'G'B'A u8"), babl_format("YA float"), "rgba_gamma_u8_to_yaf", NULL },
    { babl_format ("R'G'B'A u8"), babl_format("YA float"), "rgba_gamma_u8_to_yaf_grey", NULL },
    { babl_format ("R'G'B'A u8"), babl_format("YA float"), "rgba_gamma_u8_to_yaf_d", NULL },
    { babl_format ("R'G'B'A u8"), babl_format("YA float"), "rgba_gamma_u8_to_yaf_d_grey", NULL },
    { babl_format ("R'G'B' u8"), babl_format("RGBA float"), "rgb_gamma_u8_to_rgbaf", NULL },
    { babl_format ("R'G'B' u8"), babl_format("RaGaBaA float"), "rgb_gamma_u8_to_ragabaf", NULL },
    { babl_format ("R'G'B' u8"), babl_format("YA float"), "rgb_gamma_u8_to_yaf", NULL },
    { babl_format ("R'G'B' u8"), babl_format("YA float"), "rgb_gamma_u8_to_yaf_grey", NULL },
    { babl_format ("R'G'B' u8"), babl_format("YA float"), "rgb_gamma_u8_to_yaf_d", NULL },
    { babl_format ("R'G'B' u8"), babl_format("YA float"), "rgb_gamma_u8_to_yaf_d_grey", NULL },
  };

  /* Fail if the kernels are already compiled, meaning this must have been called twice */
  g_return_val_if_fail (!float_kernels, FALSE);
  g_return_val_if_fail (!color_kernels_hash, FALSE);

  color_kernels_hash = g_hash_table_new_full (color_kernels_hash_hashfunc,
                                              color_kernels_hash_equalfunc,
                                              NULL,
                                              NULL);

  float_source = babl_kernel_header ();
  g_string_append (float_source, colors_cl_source);

  lut8_source = babl_kernel_header ();
  for (gint i = 0; i < G_N_ELEMENTS (lut8_conversions); i++)
    {
      const gchar *from = babl_get_name (lut8_conversions[i].from_fmt);
      const gchar *to   = babl_get_name (lut8_conversions[i].to_fmt);
      gchar       *name;

      /* For YA the table can not be recovered from the result */
      if (lut8_conversions[i].to_fmt == babl_format ("YA float"))
        continue;

      name = g_strconcat (lut8_conversions[i].kernel_name, "_lut", NULL);
      append_u8_table (lut8_source, name, from, to);
      g_free (name);
    }
  append_gamma_u8_tables (lut8_source);
  g_string_append (lut8_source, colors_8bit_lut_cl_source);

  gegl_cl_color_load_conversion_set (float_conversions,
                                     G_N_ELEMENTS (float_conversions),
                                     &float_kernels,
                                     float_source->str);

  if (!float_kernels)
    {
      g_warning ("OpenCL Failed to compile color conversions (float_kernels)");
      result = FALSE;
    }

  gegl_cl_color_load_conversion_set (lut8_conversions,
                                     G_N_ELEMENTS (lut8_conversions),
                                     &lut8_kernels,
                                     lut8_source->str);

  g_string_free (float_source, TRUE);
  g_string_free (lut8_source, TRUE);

  g_hash_table_foreach (color_kernels_hash, pick_variant, NULL);

  if (!lut8_kernels)
    {
      g_warning ("OpenCL Failed to compile color conversions (lut8_kernels)");
      result = FALSE;
    }

  return result;
}

static cl_kernel
find_color_kernel (const Babl *in_format,
                   const Babl *out_format)
{
  ColorConversionInfo *info;
  ColorConversionInfo search = { in_format, out_format, NULL, NULL };
  info = g_hash_table_lookup (color_kernels_hash, &search);
  if (info)
    return info->kernel;
  return NULL;
}

gboolean
gegl_cl_color_babl (const Babl *buffer_format,
                    size_t     *bytes)
{
  if (bytes)
    *bytes = babl_format_get_bytes_per_pixel (buffer_format);

  return TRUE;
}

GeglClColorOp
gegl_cl_color_supported (const Babl *in_format,
                         const Babl *out_format)
{
  if (in_format == out_format)
    return GEGL_CL_COLOR_EQUAL;

  if (color_kernels_hash && find_color_kernel (in_format, out_format))
    return GEGL_CL_COLOR_CONVERT;

  GEGL_NOTE (GEGL_DEBUG_OPENCL, "Missing OpenCL conversion for %s -> %s",
             babl_get_name (in_format),
             babl_get_name (out_format));

  return GEGL_CL_COLOR_NOT_SUPPORTED;
}

gboolean
gegl_cl_color_conv (cl_mem         in_tex,
                    cl_mem         out_tex,
                    const size_t   size,
                    const Babl    *in_format,
                    const Babl    *out_format)
{
  cl_kernel kernel = NULL;
  cl_int cl_err = 0;

  if (in_format == out_format)
    {
      size_t s = babl_format_get_bytes_per_pixel (in_format);

      /* just copy in_tex to out_tex */
      cl_err = gegl_clEnqueueCopyBuffer (gegl_cl_get_command_queue(),
                                         in_tex, out_tex, 0, 0, size * s,
                                         0, NULL, NULL);
      CL_CHECK;
      return FALSE;
    }

  kernel = find_color_kernel (in_format, out_format);
  if (!kernel)
    return FALSE;

  return run_color_kernel (kernel, in_tex, out_tex, size);

error:
  return TRUE;
}
