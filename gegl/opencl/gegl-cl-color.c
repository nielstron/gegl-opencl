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

#include "gegl.h"
#include "gegl/gegl-debug.h"
#include "gegl-cl.h"
#include "gegl-cl-color.h"

#include "operations/opencl/colors.cl.h"
#include "operations/opencl/colors-8bit-lut.cl.h"
#include "operations/opencl/colors-8bit-quant.cl.h"

#include <string.h>

static GHashTable  *color_kernels_hash = NULL;

typedef enum
{
  TABLE_NONE,    /* not a table conversion */
  TABLE_PENDING, /* table not computed yet */
  TABLE_READY,
  TABLE_FAILED   /* the table conversion does not match babl */
} TableState;

typedef struct
{
  const Babl *from_fmt;
  const Babl *to_fmt;
  const char *kernel_name;
  cl_kernel   kernel;
  gint        table_state; /* TableState, accessed atomically */
  gint        table_kind;
} ColorConversionInfo;

static GMutex table_mutex;

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

/* Maps floats to unsigned integers with the same ordering */
static guint32
float_to_ordered (gfloat value)
{
  guint32 bits;

  memcpy (&bits, &value, sizeof (bits));

  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

static gfloat
ordered_to_float (guint32 key)
{
  guint32 bits = (key & 0x80000000u) ? (key & 0x7fffffffu) : ~key;
  gfloat  value;

  memcpy (&value, &bits, sizeof (value));

  return value;
}

/* Conversions done with tables computed from babl, see
 * colors-8bit-quant.cl.  They only work for conversions that treat each
 * channel separately, which check_table_conversion() verifies against
 * babl before a conversion is used.
 */
typedef enum
{
  TABLE_QUANTIZE_RGBA,   /* 4 float channels -> RGBA u8 */
  TABLE_QUANTIZE_RAGABA, /* premultiplied 4 float channels -> RGBA u8 */
  TABLE_QUANTIZE_YA,     /* 2 float channels -> RGBA u8 */
  TABLE_EXPAND_RGBA      /* RGBA u8 -> 4 float channels */
} TableKind;

typedef struct
{
  const gchar *from;
  const gchar *to;
  TableKind    kind;
} TableConversion;

static const TableConversion table_conversions[] = {
  { "RGBA float",    "R'G'B'A u8",    TABLE_QUANTIZE_RGBA   },
  { "R'G'B'A float", "R'G'B'A u8",    TABLE_QUANTIZE_RGBA   },
  { "R~G~B~A float", "R'G'B'A u8",    TABLE_QUANTIZE_RGBA   },
  { "RaGaBaA float", "R'G'B'A u8",    TABLE_QUANTIZE_RAGABA },
  { "YA float",      "R'G'B'A u8",    TABLE_QUANTIZE_YA     },
  { "Y'A float",     "R'G'B'A u8",    TABLE_QUANTIZE_YA     },
  { "R'G'B'A u8",    "R~G~B~A float", TABLE_EXPAND_RGBA     },
};

static const gchar *
table_kernel_name (TableKind kind)
{
  switch (kind)
    {
    case TABLE_QUANTIZE_RGBA:   return "quantize_rgbaf_to_u8";
    case TABLE_QUANTIZE_RAGABA: return "quantize_ragabaf_to_u8";
    case TABLE_QUANTIZE_YA:     return "quantize_yaf_to_u8";
    case TABLE_EXPAND_RGBA:     return "expand_rgbau8_to_f";
    }

  g_return_val_if_reached (NULL);
}

/* Fills lut[1..255] with the smallest float that babl converts to an 8-bit
 * value of at least k, for one channel of a conversion from a float format
 * to an 8-bit format (both with alpha as the last component).  With
 * alpha == TRUE the thresholds are for the alpha channel, otherwise for
 * the color channels (the alpha is then 1.0).
 */
static void
make_quantization_lut (const Babl *in_format,
                       const Babl *out_format,
                       gboolean    alpha,
                       gfloat     *lut)
{
  const Babl *fish   = babl_fish (in_format, out_format);
  const gint  n_in   = babl_format_get_n_components (in_format);
  const gint  n_out  = babl_format_get_n_components (out_format);
  const gint  out_ch = alpha ? n_out - 1 : 0;
  gfloat     *src    = g_new (gfloat, 255 * n_in);
  guint8     *dst    = g_new (guint8, 255 * n_out);
  guint32     lo[256];
  guint32     hi[256];
  gboolean    done   = FALSE;

  /* invariant: babl (lo[k]) < k <= babl (hi[k]) */
  for (gint k = 1; k < 256; k++)
    {
      lo[k] = float_to_ordered (-1.0f);
      hi[k] = float_to_ordered (2.0f);
    }

  while (! done)
    {
      done = TRUE;

      for (gint k = 1; k < 256; k++)
        {
          gfloat value = ordered_to_float (lo[k] + (hi[k] - lo[k]) / 2);
          gfloat *pixel = src + (k - 1) * n_in;

          for (gint c = 0; c < n_in - 1; c++)
            pixel[c] = alpha ? 0.0f : value;
          pixel[n_in - 1] = alpha ? value : 1.0f;
        }

      babl_process (fish, src, dst, 255);

      for (gint k = 1; k < 256; k++)
        {
          guint32 mid = lo[k] + (hi[k] - lo[k]) / 2;

          if (hi[k] - lo[k] <= 1)
            continue;

          if (dst[(k - 1) * n_out + out_ch] >= k)
            hi[k] = mid;
          else
            lo[k] = mid;

          if (hi[k] - lo[k] > 1)
            done = FALSE;
        }
    }

  lut[0] = -G_MAXFLOAT; /* never read */
  for (gint k = 1; k < 256; k++)
    lut[k] = ordered_to_float (hi[k]);

  g_free (src);
  g_free (dst);
}

/* Fills lut[0..255] with babl's float values for the 8-bit values of the
 * color channels, and lut[256..511] for those of the alpha channel.
 */
static void
make_expansion_lut (const Babl *in_format,
                    const Babl *out_format,
                    gfloat     *lut)
{
  guint8 src[256 * 4];
  gfloat dst[256 * 4];

  for (gint k = 0; k < 256; k++)
    {
      src[4 * k + 0] = src[4 * k + 1] = src[4 * k + 2] = k;
      src[4 * k + 3] = 255;
    }
  babl_process (babl_fish (in_format, out_format), src, dst, 256);
  for (gint k = 0; k < 256; k++)
    lut[k] = dst[4 * k];

  for (gint k = 0; k < 256; k++)
    {
      src[4 * k + 0] = src[4 * k + 1] = src[4 * k + 2] = 0;
      src[4 * k + 3] = k;
    }
  babl_process (babl_fish (in_format, out_format), src, dst, 256);
  for (gint k = 0; k < 256; k++)
    lut[256 + k] = dst[4 * k + 3];
}

/* Host versions of the kernels, to check the tables against babl */
static guint8
quantize_host (gfloat        value,
               const gfloat *lut)
{
  gint i = 0;

  for (gint step = 128; step > 0; step /= 2)
    if (lut[i + step] <= value)
      i += step;

  return i;
}

static void
table_convert_host (TableKind     kind,
                    const gfloat *lut,
                    const void   *in,
                    void         *out)
{
  const gfloat *src_f = in;
  const guint8 *src_u = in;
  guint8       *dst_u = out;
  gfloat       *dst_f = out;
  gfloat        alpha;

  switch (kind)
    {
    case TABLE_QUANTIZE_RGBA:
      for (gint c = 0; c < 3; c++)
        dst_u[c] = quantize_host (src_f[c], lut);
      dst_u[3] = quantize_host (src_f[3], lut + 256);
      break;

    case TABLE_QUANTIZE_RAGABA:
      alpha = src_f[3];
      if (alpha < 1.0f / 65536.0f)
        {
          if (alpha >= 0.0f)
            alpha = 1.0f / 65536.0f;
          else if (alpha >= -1.0f / 65536.0f)
            alpha = -1.0f / 65536.0f;
        }
      alpha = 1.0f / alpha;
      for (gint c = 0; c < 3; c++)
        dst_u[c] = quantize_host (src_f[c] * alpha, lut);
      dst_u[3] = quantize_host (src_f[3], lut + 256);
      break;

    case TABLE_QUANTIZE_YA:
      dst_u[0] = dst_u[1] = dst_u[2] = quantize_host (src_f[0], lut);
      dst_u[3] = quantize_host (src_f[1], lut + 256);
      break;

    case TABLE_EXPAND_RGBA:
      for (gint c = 0; c < 3; c++)
        dst_f[c] = lut[src_u[c]];
      dst_f[3] = lut[256 + src_u[3]];
      break;
    }
}

/* Compares the table conversion with babl on a set of pixels.  Converting
 * to 8-bit may be off by one at ties, as babl's own result can depend on
 * the conversion path it picked (for example when it goes through a
 * premultiplied format), anything else rejects the conversion.
 */
static gboolean
check_table_conversion (const Babl   *in_format,
                        const Babl   *out_format,
                        TableKind     kind,
                        const gfloat *lut)
{
  const gint   n          = 8192;
  const gint   in_bpp     = babl_format_get_bytes_per_pixel (in_format);
  const gint   out_bpp    = babl_format_get_bytes_per_pixel (out_format);
  const gint   n_in       = babl_format_get_n_components (in_format);
  guint8      *src        = g_malloc (n * in_bpp);
  guint8      *ref        = g_malloc (n * out_bpp);
  guint8       out[16];
  guint32      seed       = 1;
  gboolean     ok         = TRUE;

  for (gint i = 0; i < n; i++)
    {
      for (gint c = 0; c < n_in; c++)
        {
          seed = seed * 1664525u + 1013904223u;

          if (kind == TABLE_EXPAND_RGBA)
            {
              src[i * in_bpp + c] = seed >> 24;
            }
          else
            {
              gfloat *pixel = (gfloat *) (src + i * in_bpp);
              gfloat  value = (seed >> 8) / (gfloat) (1 << 24) * 1.1f - 0.05f;

              if (c == n_in - 1)
                value = (i % 2) ? 1.0f : CLAMP (value, 0.0f, 1.0f);

              pixel[c] = value;
            }
        }

      if (kind == TABLE_QUANTIZE_RAGABA)
        {
          gfloat *pixel = (gfloat *) (src + i * in_bpp);

          for (gint c = 0; c < 3; c++)
            pixel[c] *= pixel[3];
        }
    }

  babl_process (babl_fish (in_format, out_format), src, ref, n);

  for (gint i = 0; i < n && ok; i++)
    {
      table_convert_host (kind, lut, src + i * in_bpp, out);

      for (gint c = 0; c < 4 && ok; c++)
        {
          if (kind == TABLE_EXPAND_RGBA)
            ok = ((gfloat *) out)[c] == ((gfloat *) (ref + i * out_bpp))[c];
          else
            ok = ABS (out[c] - ref[i * out_bpp + c]) <= 1;
        }
    }

  g_free (src);
  g_free (ref);

  return ok;
}

/* Computes the table of a table conversion and binds it to the kernel.
 * This is done on first use, as building babl's fish can take a while.
 */
static gboolean
setup_table_conversion (ColorConversionInfo *info)
{
  TableKind kind = info->table_kind;
  gfloat    lut[512];
  cl_mem    lut_mem;
  cl_int    cl_err = 0;

  g_mutex_lock (&table_mutex);

  if (info->table_state != TABLE_PENDING)
    {
      g_mutex_unlock (&table_mutex);
      return info->table_state == TABLE_READY;
    }

  if (kind == TABLE_EXPAND_RGBA)
    {
      make_expansion_lut (info->from_fmt, info->to_fmt, lut);
    }
  else
    {
      make_quantization_lut (info->from_fmt, info->to_fmt, FALSE, lut);
      make_quantization_lut (info->from_fmt, info->to_fmt, TRUE, lut + 256);
    }

  if (! check_table_conversion (info->from_fmt, info->to_fmt, kind, lut))
    {
      GEGL_NOTE (GEGL_DEBUG_OPENCL,
                 "Not using OpenCL table conversion %s -> %s, "
                 "it does not match babl",
                 babl_get_name (info->from_fmt),
                 babl_get_name (info->to_fmt));
      g_atomic_int_set (&info->table_state, TABLE_FAILED);
      g_mutex_unlock (&table_mutex);
      return FALSE;
    }

  lut_mem = gegl_clCreateBuffer (gegl_cl_get_context (),
                                 CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 sizeof (lut), lut, &cl_err);
  if (cl_err == CL_SUCCESS)
    {
      /* the table stays bound to the kernel, which is only used for this
       * conversion
       */
      cl_err = gegl_clSetKernelArg (info->kernel, 2, sizeof (cl_mem), &lut_mem);

      if (cl_err != CL_SUCCESS)
        gegl_clReleaseMemObject (lut_mem);
    }

  g_atomic_int_set (&info->table_state,
                    cl_err == CL_SUCCESS ? TABLE_READY : TABLE_FAILED);

  g_mutex_unlock (&table_mutex);

  return cl_err == CL_SUCCESS;
}

static gboolean
gegl_cl_color_load_table_conversions (GeglClRunData **kernels,
                                      const gchar    *source)
{
  const gint    n_conversions = G_N_ELEMENTS (table_conversions);
  const char  **kernel_names  = g_newa (const char *, n_conversions + 1);

  for (gint i = 0; i < n_conversions; i++)
    kernel_names[i] = table_kernel_name (table_conversions[i].kind);

  kernel_names[n_conversions] = NULL;

  /* every conversion gets its own kernel object, holding its own table */
  *kernels = gegl_cl_compile_and_build (source, kernel_names);

  if (!(*kernels))
    return FALSE;

  for (gint i = 0; i < n_conversions; i++)
    {
      ColorConversionInfo *info = g_new0 (ColorConversionInfo, 1);

      info->from_fmt    = babl_format (table_conversions[i].from);
      info->to_fmt      = babl_format (table_conversions[i].to);
      info->kernel_name = kernel_names[i];
      info->kernel      = (*kernels)->kernel[i];
      info->table_state = TABLE_PENDING;
      info->table_kind  = table_conversions[i].kind;

      g_hash_table_insert (color_kernels_hash, info, info);
    }

  return TRUE;
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
      ColorConversionInfo *info = g_new (ColorConversionInfo, 1);

      conversions[i].kernel = (*kernels)->kernel[i];
      *info = conversions[i];

      g_hash_table_insert (color_kernels_hash, info, info);
    }

  return TRUE;
}

gboolean
gegl_cl_color_compile_kernels (void)
{
  static GeglClRunData *float_kernels = NULL;
  static GeglClRunData *lut8_kernels = NULL;
  static GeglClRunData *table_kernels = NULL;
  gboolean result = TRUE;

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

    { babl_format ("RGBA float"), babl_format("Y'CbCrA float"), "rgbaf_to_ycbcraf", NULL },
    { babl_format ("Y'CbCrA float"), babl_format("RGBA float"), "ycbcraf_to_rgbaf", NULL },
    { babl_format ("RGBA u8"), babl_format("Y'CbCrA float"), "rgbau8_to_ycbcraf", NULL },
    { babl_format ("Y'CbCrA float"), babl_format("RGBA u8"), "ycbcraf_to_rgbau8", NULL },

    { babl_format ("RGB u8"), babl_format("RGBA float"), "rgbu8_to_rgbaf", NULL },
    { babl_format ("RGBA float"), babl_format("RGB u8"), "rgbaf_to_rgbu8", NULL },

    { babl_format ("Y u8"), babl_format("Y float"), "yu8_to_yf", NULL },

    { babl_format ("RGBA float"), babl_format("YA float"), "rgbaf_to_yaf", NULL },
    { babl_format ("YA float"), babl_format("RGBA float"), "yaf_to_rgbaf", NULL },
    { babl_format ("RGBA u8"), babl_format("YA float"), "rgbau8_to_yaf", NULL },
    { babl_format ("YA float"), babl_format("RGBA u8"), "yaf_to_rgbau8", NULL },
    { babl_format ("RaGaBaA float"), babl_format("YA float"), "ragabaf_to_yaf", NULL },

    { babl_format ("RGBA float"), babl_format("R'G'B' u8"), "rgbaf_to_rgb_gamma_u8", NULL },

    { babl_format ("RaGaBaA float"), babl_format("R'G'B' u8"), "ragabaf_to_rgb_gamma_u8", NULL },

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
    { babl_format ("R'G'B' u8"), babl_format("RGBA float"), "rgb_gamma_u8_to_rgbaf", NULL },
    { babl_format ("R'G'B' u8"), babl_format("RaGaBaA float"), "rgb_gamma_u8_to_ragabaf", NULL },
    { babl_format ("R'G'B' u8"), babl_format("YA float"), "rgb_gamma_u8_to_yaf", NULL },
  };

  /* Fail if the kernels are already compiled, meaning this must have been called twice */
  g_return_val_if_fail (!float_kernels, FALSE);
  g_return_val_if_fail (!color_kernels_hash, FALSE);

  color_kernels_hash = g_hash_table_new_full (color_kernels_hash_hashfunc,
                                              color_kernels_hash_equalfunc,
                                              NULL,
                                              NULL);

  gegl_cl_color_load_conversion_set (float_conversions,
                                     G_N_ELEMENTS (float_conversions),
                                     &float_kernels,
                                     colors_cl_source);

  if (!float_kernels)
    {
      g_warning ("OpenCL Failed to compile color conversions (float_kernels)");
      result = FALSE;
    }

  gegl_cl_color_load_conversion_set (lut8_conversions,
                                     G_N_ELEMENTS (lut8_conversions),
                                     &lut8_kernels,
                                     colors_8bit_lut_cl_source);

  if (!lut8_kernels)
    {
      g_warning ("OpenCL Failed to compile color conversions (lut8_kernels)");
      result = FALSE;
    }

  gegl_cl_color_load_table_conversions (&table_kernels,
                                        colors_8bit_quant_cl_source);

  if (!table_kernels)
    {
      g_warning ("OpenCL Failed to compile color conversions (table_kernels)");
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
  if (! info)
    return NULL;
  if (g_atomic_int_get (&info->table_state) != TABLE_NONE &&
      g_atomic_int_get (&info->table_state) != TABLE_READY &&
      ! setup_table_conversion (info))
    return NULL;
  return info->kernel;
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
