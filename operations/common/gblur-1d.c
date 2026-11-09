/* This file is an image processing operation for GEGL
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
 * Copyright 2006 Dominik Ernst <dernst@gmx.de>
 * Copyright 2013 Massimo Valentini <mvalentini@src.gnome.org>
 *     2017, 2018 Øyvind Kolås <pippin@gimp.org>
 *
 * Recursive Gauss IIR Filter as described by Young / van Vliet
 * in "Signal Processing 44 (1995) 139 - 151"
 *
 * NOTE: The IIR filter should not be used for radius < 0.5, since it
 *       becomes very inaccurate.
 */

#include "config.h"
#include <glib/gi18n-lib.h>

#ifdef GEGL_PROPERTIES

enum_start (gegl_gblur_1d_policy)
   enum_value (GEGL_GBLUR_1D_ABYSS_NONE,  "none",  N_("None"))
   enum_value (GEGL_GBLUR_1D_ABYSS_CLAMP, "clamp", N_("Clamp"))
   enum_value (GEGL_GBLUR_1D_ABYSS_BLACK, "black", N_("Black"))
   enum_value (GEGL_GBLUR_1D_ABYSS_WHITE, "white", N_("White"))
enum_end (GeglGblur1dPolicy)

enum_start (gegl_gblur_1d_filter)
  enum_value (GEGL_GBLUR_1D_AUTO, "auto", N_("Auto"))
  enum_value (GEGL_GBLUR_1D_FIR,  "fir",  N_("FIR"))
  enum_value (GEGL_GBLUR_1D_IIR,  "iir",  N_("IIR"))
enum_end (GeglGblur1dFilter)

property_double (std_dev, _("Size"), 1.5)
  description (_("Standard deviation (spatial scale factor)"))
  value_range   (0.0, 1500.0)
  ui_range      (0.0, 100.0)
  ui_gamma      (3.0)

property_enum (orientation, _("Orientation"),
               GeglOrientation, gegl_orientation,
               GEGL_ORIENTATION_HORIZONTAL)
  description (_("The orientation of the blur - hor/ver"))

property_enum (filter, _("Filter"),
               GeglGblur1dFilter, gegl_gblur_1d_filter,
               GEGL_GBLUR_1D_AUTO)
  description (_("How the gaussian kernel is discretized"))

property_enum (abyss_policy, _("Abyss policy"), GeglGblur1dPolicy,
               gegl_gblur_1d_policy, GEGL_GBLUR_1D_ABYSS_NONE)
  description (_("How image edges are handled"))

property_boolean (clip_extent, _("Clip to the input extent"), TRUE)
  description (_("Should the output extent be clipped to the input extent"))

#else

#define GEGL_OP_FILTER
#define GEGL_OP_NAME     gblur_1d
#define GEGL_OP_C_SOURCE gblur-1d.c

#include "gegl-op.h"

/**********************************************
 *
 * Infinite Impulse Response (IIR)
 *
 **********************************************/

typedef void (*IirYoungBlur1dFunc) (gfloat           *buf,
                                    gdouble          *tmp,
                                    const gdouble    *b,
                                    gdouble         (*m)[3],
                                    const gfloat     *iminus,
                                    const gfloat     *uplus,
                                    const gint        len,
                                    const gint        components,
                                    GeglAbyssPolicy   policy);

static const gfloat white[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
static const gfloat black[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
static const gfloat none[4]  = { 0.0f, 0.0f, 0.0f, 0.0f };

static void
iir_young_find_constants (gfloat   sigma,
                          gdouble *b,
                          gdouble (*m)[3])
{
  const gdouble K1 = 2.44413;
  const gdouble K2 = 1.4281;
  const gdouble K3 = 0.422205;
  const gdouble q = sigma >= 2.5 ?
                    0.98711 * sigma - 0.96330 :
                    3.97156 - 4.14554 * sqrt (1 - 0.26891 * sigma);

  const gdouble b0 = 1.57825 + q*(K1 + q*(    K2 + q *     K3));
  const gdouble b1 =           q*(K1 + q*(2 * K2 + q * 3 * K3));
  const gdouble b2 =     (-K2 * q * q) + (-K3 * 3 * q * q * q);
  const gdouble b3 =           q*      q*          q *     K3;

  const gdouble a1 = b1 / b0;
  const gdouble a2 = b2 / b0;
  const gdouble a3 = b3 / b0;

  const gdouble c  = 1. / ((1+a1-a2+a3) * (1+a2+(a1-a3)*a3));

  m[0][0] = c * (-a3*(a1+a3)-a2 + 1);
  m[0][1] = c * (a3+a1)*(a2+a3*a1);
  m[0][2] = c * a3*(a1+a3*a2);

  m[1][0] = c * (a1+a3*a2);
  m[1][1] = c * (1-a2)*(a2+a3*a1);
  m[1][2] = c * a3*(1-a3*a1-a3*a3-a2);

  m[2][0] = c * (a3*a1+a2+a1*a1-a2*a2);
  m[2][1] = c * (a1*a2+a3*a2*a2-a1*a3*a3-a3*a3*a3-a3*a2+a3);
  m[2][2] = c * a3*(a1+a3*a2);

  b[0] = 1. - (b1 + b2 + b3) / b0;
  b[1] = a1;
  b[2] = a2;
  b[3] = a3;
}

static void
get_boundaries (GeglAbyssPolicy   policy,
                gfloat           *buf,
                gint              len,
                gint              nc,
                const gfloat    **out_iminus,
                const gfloat    **out_uplus)
{
  const gfloat *iminus;
  const gfloat *uplus;

  switch (policy)
    {
    case GEGL_ABYSS_CLAMP:
    default:
      iminus = &buf[nc * 3]; uplus = &buf[nc * (len + 2)];
      break;

    case GEGL_ABYSS_NONE:
      iminus = uplus = &none[0];
      break;

    case GEGL_ABYSS_WHITE:
      iminus = uplus = &white[0];
      break;

    case GEGL_ABYSS_BLACK:
      iminus = uplus = &black[nc == 2 ? 2 : 0];
      break;
    }

  if (out_iminus != NULL)
    *out_iminus = iminus;

  if (out_uplus != NULL)
    *out_uplus = uplus;
}

static inline void
fix_right_boundary_y (gdouble        *buf,
                      gdouble       (*m)[3],
                      const gfloat   *uplus)
{
  gdouble u[3] = { buf[-1] - uplus[0],
                   buf[-2] - uplus[0],
                   buf[-3] - uplus[0] };
  gint    i, k;

  for (i = 0; i < 3; i++)
    {
      gdouble tmp = 0.;

      for (k = 0; k < 3; k++)
        tmp += m[i][k] * u[k];

      buf[i] = tmp + uplus[0];
    }
}

static void
iir_young_blur_1D_y (gfloat        *buf,
                     gdouble       *tmp,
                     const gdouble *b,
                     gdouble      (*m)[3],
                     const gfloat  *iminus,
                     const gfloat  *uplus,
                     const gint     len,
                     const gint     components,
                     GeglAbyssPolicy policy)
{
  gint    i, j;

  for (i = 0; i < 3; i++, tmp++)
    tmp[0] = iminus[0];

  buf += 3;

  for (i = 3; i < 3 + len; i++, buf++, tmp++)
    {
      tmp[0] = b[0] * buf[0];

      for (j = 1; j < 4; ++j)
        {
          gint offset = -1 * j;
          tmp[0] += b[j] * tmp[offset + 0];
        }
    }

  fix_right_boundary_y (tmp, m, uplus);

  buf--;
  tmp--;

  for (i = 3 + len - 1; 3 <= i; i--, buf--, tmp--)
    {
      tmp[0] *= b[0];

      for (j = 1; j < 4; ++j)
        {
          gint offset = j;
          tmp[0] += b[j] * tmp[offset + 0];
        }

      buf[0] = tmp[0];
    }
}

static inline void
fix_right_boundary_yA (gdouble        *buf,
                       gdouble       (*m)[3],
                       const gfloat   *uplus)
{
  gdouble u[6] = { buf[-2] - uplus[0], buf[-1] - uplus[1],
                   buf[-4] - uplus[0], buf[-3] - uplus[1],
                   buf[-6] - uplus[0], buf[-5] - uplus[1] };
  gint    i, k;

  for (i = 0; i < 3; i++)
    {
      gdouble tmp[2] = { 0.0, 0.0 };

      for (k = 0; k < 3; k++)
        {
          tmp[0] += m[i][k] * u[k * 2 + 0];
          tmp[1] += m[i][k] * u[k * 2 + 1];
        }

      buf[2 * i + 0] = tmp[0] + uplus[0];
      buf[2 * i + 1] = tmp[1] + uplus[1];
    }
}

static void
iir_young_blur_1D_yA (gfloat           *buf,
                      gdouble          *tmp,
                      const gdouble    *b,
                      gdouble         (*m)[3],
                      const gfloat     *iminus,
                      const gfloat     *uplus,
                      const gint        len,
                      const gint        components,
                      GeglAbyssPolicy   policy)
{
  gint    i, j;

  for (i = 0; i < 3; i++, tmp += 2)
    {
      tmp[0] = iminus[0];
      tmp[1] = iminus[1];
    }

  buf += 6;

  for (i = 3; i < 3 + len; i++, buf += 2, tmp += 2)
    {
      tmp[0] = b[0] * buf[0];
      tmp[1] = b[0] * buf[1];

      for (j = 1; j < 4; ++j)
        {
          gint offset = -2 * j;

          tmp[0] += b[j] * tmp[offset + 0];
          tmp[1] += b[j] * tmp[offset + 1];
        }
    }

  fix_right_boundary_yA (tmp, m, uplus);

  buf -= 2;
  tmp -= 2;

  for (i = 3 + len - 1; 3 <= i; i--, buf -= 2, tmp -= 2)
    {
      tmp[0] *= b[0];
      tmp[1] *= b[0];

      for (j = 1; j < 4; ++j)
        {
          gint offset = 2 * j;

          tmp[0] += b[j] * tmp[offset + 0];
          tmp[1] += b[j] * tmp[offset + 1];
        }

      buf[0] = tmp[0];
      buf[1] = tmp[1];
    }
}


static inline void
fix_right_boundary_generic (gdouble        *buf,
                            gdouble       (*m)[3],
                            const gfloat   *uplus,
                            const gint      nc)
{
  gdouble *u   = g_newa (gdouble, nc * 3);
  gdouble *tmp = g_newa (gdouble, nc);
  gint    i, k, c;

  for (k = 0; k < 3; k++)
    for (c = 0; c < nc; c++)
   u[k * nc + c] = buf[(-k-1)*nc+c] - uplus[c];

  for (i = 0; i < 3; i++)
    {
      for (c = 0; c < nc ; c++)
        tmp[c] = m[i][0] * u[0 * nc + c];

      for (k = 1; k < 3; k++)
        {
          for (c = 0; c < nc ; c++)
            tmp[c] += m[i][k] * u[k * nc + c];
        }

      for (c = 0; c < nc ; c++)
        buf[nc * i + c] = tmp[c] + uplus[c];
    }
}

static void
iir_young_blur_1D_generic (gfloat           *buf,
                           gdouble          *tmp,
                           const gdouble    *b,
                           gdouble         (*m)[3],
                           const gfloat     *iminus,
                           const gfloat     *uplus,
                           const gint        len,
                           const gint        components,
                           GeglAbyssPolicy   policy)
{
  gint    i, j, c;
  for (i = 0; i < 3; i++, tmp += components)
    {
      for (c = 0; c < components; c++)
        tmp[c] = iminus[c];
    }

  buf += 3 * components;

  for (i = 0; i < len; i++, buf += components, tmp += components)
    {
      for (c = 0; c < components; c++)
        tmp[c] = b[0] * buf[c];

      for (j = 1; j < 4; ++j)
        {
          gint offset = -components * j;

          for (c = 0; c < components; c++)
            tmp[c] += b[j] * tmp[offset + c];
        }
    }

  fix_right_boundary_generic (tmp, m, uplus, components);

  buf -= components;
  tmp -= components;

  for (i = 3 + len - 1; 3 <= i; i--, buf -= components, tmp -= components)
    {
      for (c = 0; c < components; c++)
        tmp[c] *= b[0];

      for (j = 1; j < 4; ++j)
        {
          gint offset = components * j;

          for (c = 0; c < components; c++)
            tmp[c] += b[j] * tmp[offset + c];
        }

      for (c = 0; c < components; c++)
        buf[c] = tmp[c];
    }
}

static inline void
fix_right_boundary_rgb (gdouble        *buf,
                        gdouble       (*m)[3],
                        const gfloat   *uplus)
{
  gdouble u[9] = { buf[-3] - uplus[0], buf[-2] - uplus[1], buf[-1] - uplus[2],
                   buf[-6] - uplus[0], buf[-5] - uplus[1], buf[-4] - uplus[2],
                   buf[-9] - uplus[0], buf[-8] - uplus[1], buf[-7] - uplus[2] };
  gint    i, k;

  for (i = 0; i < 3; i++)
    {
      gdouble tmp[3] = { 0.0, 0.0, 0.0 };

      for (k = 0; k < 3; k++)
        {
          tmp[0] += m[i][k] * u[k * 3 + 0];
          tmp[1] += m[i][k] * u[k * 3 + 1];
          tmp[2] += m[i][k] * u[k * 3 + 2];
        }

      buf[3 * i + 0] = tmp[0] + uplus[0];
      buf[3 * i + 1] = tmp[1] + uplus[1];
      buf[3 * i + 2] = tmp[2] + uplus[2];
    }
}

static void
iir_young_blur_1D_rgb (gfloat           *buf,
                       gdouble          *tmp,
                       const gdouble    *b,
                       gdouble         (*m)[3],
                       const gfloat     *iminus,
                       const gfloat     *uplus,
                       const gint        len,
                       const gint        components,
                       GeglAbyssPolicy   policy)
{
  gint    i, j;

  for (i = 0; i < 3; i++, tmp += 3)
    {
      tmp[0] = iminus[0];
      tmp[1] = iminus[1];
      tmp[2] = iminus[2];
    }

  buf += 9;

  for (i = 3; i < 3 + len; i++, buf += 3, tmp += 3)
    {
      tmp[0] = b[0] * buf[0];
      tmp[1] = b[0] * buf[1];
      tmp[2] = b[0] * buf[2];

      for (j = 1; j < 4; ++j)
        {
          gint offset = -3 * j;

          tmp[0] += b[j] * tmp[offset + 0];
          tmp[1] += b[j] * tmp[offset + 1];
          tmp[2] += b[j] * tmp[offset + 2];
        }
    }

  fix_right_boundary_rgb (tmp, m, uplus);

  buf -= 3;
  tmp -= 3;

  for (i = 3 + len - 1; 3 <= i; i--, buf -= 3, tmp -= 3)
    {
      tmp[0] *= b[0];
      tmp[1] *= b[0];
      tmp[2] *= b[0];

      for (j = 1; j < 4; ++j)
        {
          gint offset = 3 * j;

          tmp[0] += b[j] * tmp[offset + 0];
          tmp[1] += b[j] * tmp[offset + 1];
          tmp[2] += b[j] * tmp[offset + 2];
        }

      buf[0] = tmp[0];
      buf[1] = tmp[1];
      buf[2] = tmp[2];
    }
}

static inline void
fix_right_boundary_rgbA (gdouble        *buf,
                         gdouble       (*m)[3],
                         const gfloat   *uplus)
{
  gdouble u[12] = { buf[-4] - uplus[0], buf[-3] - uplus[1], buf[-2] - uplus[2], buf[-1] - uplus[3],
                    buf[-8] - uplus[0], buf[-7] - uplus[1], buf[-6] - uplus[2], buf[-5] - uplus[3],
                    buf[-12] - uplus[0], buf[-11] - uplus[1], buf[-10] - uplus[2], buf[-9] - uplus[3] };
  gint    i, k;

  for (i = 0; i < 3; i++)
    {
      gdouble tmp[4] = { 0.0, 0.0, 0.0, 0.0 };

      for (k = 0; k < 3; k++)
        {
          tmp[0] += m[i][k] * u[k * 4 + 0];
          tmp[1] += m[i][k] * u[k * 4 + 1];
          tmp[2] += m[i][k] * u[k * 4 + 2];
          tmp[3] += m[i][k] * u[k * 4 + 3];
        }

      buf[4 * i + 0] = tmp[0] + uplus[0];
      buf[4 * i + 1] = tmp[1] + uplus[1];
      buf[4 * i + 2] = tmp[2] + uplus[2];
      buf[4 * i + 3] = tmp[3] + uplus[3];
    }
}

static void
iir_young_blur_1D_rgbA (gfloat           *buf,
                        gdouble          *tmp,
                        const gdouble    *b,
                        gdouble         (*m)[3],
                        const gfloat     *iminus,
                        const gfloat     *uplus,
                        const gint        len,
                        const gint        components,
                        GeglAbyssPolicy   policy)
{
  gint     i, j;

  for (i = 0; i < 3; i++, tmp += 4)
    {
      tmp[0] = iminus[0];
      tmp[1] = iminus[1];
      tmp[2] = iminus[2];
      tmp[3] = iminus[3];
    }

  buf += 12;

  for (i = 0; i < len; i++, buf += 4, tmp += 4)
    {
      tmp[0] = b[0] * buf[0];
      tmp[1] = b[0] * buf[1];
      tmp[2] = b[0] * buf[2];
      tmp[3] = b[0] * buf[3];

      for (j = 1; j < 4; ++j)
        {
          gint offset = -4 * j;

          tmp[0] += b[j] * tmp[offset + 0];
          tmp[1] += b[j] * tmp[offset + 1];
          tmp[2] += b[j] * tmp[offset + 2];
          tmp[3] += b[j] * tmp[offset + 3];
        }
    }

  fix_right_boundary_rgbA (tmp, m, uplus);

  buf -= 4;
  tmp -= 4;

  for (i = 3 + len - 1; 3 <= i; i--, buf -= 4, tmp -= 4)
    {
      tmp[0] *= b[0];
      tmp[1] *= b[0];
      tmp[2] *= b[0];
      tmp[3] *= b[0];

      for (j = 1; j < 4; ++j)
        {
          gint offset = 4 * j;

          tmp[0] += b[j] * tmp[offset + 0];
          tmp[1] += b[j] * tmp[offset + 1];
          tmp[2] += b[j] * tmp[offset + 2];
          tmp[3] += b[j] * tmp[offset + 3];
        }

      buf[0] = tmp[0];
      buf[1] = tmp[1];
      buf[2] = tmp[2];
      buf[3] = tmp[3];
    }
}

static void
iir_young_hor_blur (IirYoungBlur1dFunc   real_blur_1D,
                    GeglBuffer          *src,
                    const GeglRectangle *rect,
                    GeglBuffer          *dst,
                    const gdouble       *b,
                    gdouble            (*m)[3],
                    GeglAbyssPolicy      policy,
                    const Babl          *format,
                    gint                 level)
{
  GeglRectangle  cur_row = *rect;
  const gint     nc = babl_format_get_n_components (format);
  gfloat        *row = g_new (gfloat, (3 + rect->width + 3) * nc);
  gdouble       *tmp = g_new (gdouble, (3 + rect->width + 3) * nc);
  gint           v;

  cur_row.height = 1;

  for (v = 0; v < rect->height; v++)
    {
      const gfloat *iminus;
      const gfloat *uplus;

      cur_row.y = rect->y + v;

      gegl_buffer_get (src, &cur_row, 1.0/(1<<level), format, &row[3 * nc],
                       GEGL_AUTO_ROWSTRIDE, GEGL_ABYSS_NONE);

      get_boundaries (policy, row, rect->width, nc, &iminus, &uplus);
      real_blur_1D (row, tmp, b, m, iminus, uplus, rect->width, nc, policy);

      gegl_buffer_set (dst, &cur_row, level, format, &row[3 * nc],
                       GEGL_AUTO_ROWSTRIDE);
    }

  g_free (tmp);
  g_free (row);
}

static void
iir_young_ver_blur (IirYoungBlur1dFunc   real_blur_1D,
                    GeglBuffer          *src,
                    const GeglRectangle *rect,
                    GeglBuffer          *dst,
                    const gdouble       *b,
                    gdouble            (*m)[3],
                    GeglAbyssPolicy      policy,
                    const Babl          *format,
                    gint                 level)
{
  GeglRectangle  cur_col = *rect;
  const gint     nc = babl_format_get_n_components (format);
  gfloat        *col = g_new (gfloat, (3 + rect->height + 3) * nc);
  gdouble       *tmp = g_new (gdouble, (3 + rect->height + 3) * nc);
  gint           i;

  cur_col.width = 1;

  for (i = 0; i < rect->width; i++)
    {
      const gfloat *iminus;
      const gfloat *uplus;

      cur_col.x = rect->x + i;

      gegl_buffer_get (src, &cur_col, 1.0/(1<<level), format, &col[3 * nc],
                       GEGL_AUTO_ROWSTRIDE, GEGL_ABYSS_NONE);

      get_boundaries (policy, col, rect->height, nc, &iminus, &uplus);
      real_blur_1D (col, tmp, b, m, iminus, uplus, rect->height, nc, policy);

      gegl_buffer_set (dst, &cur_col, level, format, &col[3 * nc],
                       GEGL_AUTO_ROWSTRIDE);
    }

  g_free (tmp);
  g_free (col);
}


/**********************************************
 *
 * Finite Impulse Response (FIR)
 *
 **********************************************/

static inline void
fir_blur_1D (const gfloat *input,
                   gfloat *output,
             const gfloat *cmatrix,
             const gint    clen,
             const gint    len,
             const gint    nc)
{
  gint i;
  for (i = 0; i < len; i++)
    {
      gint c;
      for (c = 0; c < nc; c++)
        {
          gint   index = i * nc + c;
          gfloat acc   = 0.0f;
          gint   m;

          for (m = 0; m < clen; m++)
            acc += input [index + m * nc] * cmatrix [m];

          output [index] = acc;
        }
    }
}

static void
fir_hor_blur (GeglBuffer          *src,
              const GeglRectangle *rect,
              GeglBuffer          *dst,
              gfloat              *cmatrix,
              gint                 clen,
              GeglAbyssPolicy      policy,
              const Babl          *format,
              gint                 level)
{
  GeglRectangle  cur_row = *rect;
  GeglRectangle  in_row;
  const gint     nc = babl_format_get_n_components (format);
  gfloat        *row;
  gfloat        *out;
  gint           v;

  cur_row.height = 1;

  in_row         = cur_row;
  in_row.width  += clen - 1;
  in_row.x      -= clen / 2;

  row = gegl_malloc (sizeof (gfloat) * in_row.width  * nc);
  out = gegl_malloc (sizeof (gfloat) * cur_row.width * nc);

  for (v = 0; v < rect->height; v++)
    {
      cur_row.y = in_row.y = rect->y + v;

      gegl_buffer_get (src, &in_row, 1.0/(1<<level), format, row, GEGL_AUTO_ROWSTRIDE, policy);

      fir_blur_1D (row, out, cmatrix, clen, rect->width, nc);

      gegl_buffer_set (dst, &cur_row, level, format, out, GEGL_AUTO_ROWSTRIDE);
    }

  gegl_free (out);
  gegl_free (row);
}

static void
fir_ver_blur (GeglBuffer          *src,
              const GeglRectangle *rect,
              GeglBuffer          *dst,
              gfloat              *cmatrix,
              gint                 clen,
              GeglAbyssPolicy      policy,
              const Babl          *format,
              gint                 level)
{
  GeglRectangle  cur_col = *rect;
  GeglRectangle  in_col;
  const gint     nc = babl_format_get_n_components (format);
  gfloat        *col;
  gfloat        *out;
  gint           v;

  cur_col.width = 1;

  in_col         = cur_col;
  in_col.height += clen - 1;
  in_col.y      -= clen / 2;

  col = gegl_malloc (sizeof (gfloat) * in_col.height  * nc);
  out = gegl_malloc (sizeof (gfloat) * cur_col.height * nc);

  for (v = 0; v < rect->width; v++)
    {
      cur_col.x = in_col.x = rect->x + v;

      gegl_buffer_get (src, &in_col, 1.0/(1<<level), format, col, GEGL_AUTO_ROWSTRIDE, policy);

      fir_blur_1D (col, out, cmatrix, clen, rect->height, nc);

      gegl_buffer_set (dst, &cur_col, level, format, out, GEGL_AUTO_ROWSTRIDE);
    }

  gegl_free (out);
  gegl_free (col);
}


#include "opencl/gegl-cl.h"
#include "gegl-buffer-cl-iterator.h"

#include "opencl/gblur-1d.cl.h"

static GeglClRunData *cl_data = NULL;

static void
gblur_1d_cl_build (void)
{
  if (!cl_data)
    {
      const char *kernel_name[] = {"fir_ver_blur", "fir_hor_blur",
                                   "iir_young_causal", "iir_young_anticausal",
                                   NULL};
      cl_data = gegl_cl_compile_and_build (gblur_1d_cl_source, kernel_name);
    }
}


static gboolean
cl_gaussian_blur (cl_mem                 in_tex,
                  cl_mem                 out_tex,
                  const GeglRectangle   *roi,
                  cl_mem                 cl_cmatrix,
                  gint                   clen,
                  GeglOrientation        orientation)
{
  cl_int cl_err = 0;
  size_t global_ws[2];
  gint   kernel_num;

  gblur_1d_cl_build ();

  if (!cl_data)
    return TRUE;

  if (orientation == GEGL_ORIENTATION_VERTICAL)
    kernel_num = 0;
  else
    kernel_num = 1;

  global_ws[0] = roi->width;
  global_ws[1] = roi->height;

  cl_err = gegl_cl_set_kernel_args (cl_data->kernel[kernel_num],
                                    sizeof(cl_mem), (void*)&in_tex,
                                    sizeof(cl_mem), (void*)&out_tex,
                                    sizeof(cl_mem), (void*)&cl_cmatrix,
                                    sizeof(cl_int), (void*)&clen,
                                    NULL);
  CL_CHECK;

  cl_err = gegl_clEnqueueNDRangeKernel (gegl_cl_get_command_queue (),
                                        cl_data->kernel[kernel_num], 2,
                                        NULL, global_ws, NULL,
                                        0, NULL, NULL);
  CL_CHECK;

  cl_err = gegl_clFinish (gegl_cl_get_command_queue ());
  CL_CHECK;

  return FALSE;

error:
  return TRUE;
}

static gboolean
fir_cl_process (GeglBuffer            *input,
                GeglBuffer            *output,
                const GeglRectangle   *result,
                const Babl            *format,
                gfloat                *cmatrix,
                gint                   clen,
                GeglOrientation        orientation,
                GeglAbyssPolicy        abyss)
{
  gboolean              err = FALSE;
  cl_int                cl_err;
  cl_mem                cl_cmatrix = NULL;
  GeglBufferClIterator *i;
  gint                  read;
  gint                  left, right, top, bottom;

  if (orientation == GEGL_ORIENTATION_HORIZONTAL)
    {
      right = left = clen / 2;
      top = bottom = 0;
    }
  else
    {
      right = left = 0;
      top = bottom = clen / 2;
    }

  i = gegl_buffer_cl_iterator_new (output,
                                   result,
                                   format,
                                   GEGL_CL_BUFFER_WRITE);

  read = gegl_buffer_cl_iterator_add_2 (i,
                                        input,
                                        result,
                                        format,
                                        GEGL_CL_BUFFER_READ,
                                        left, right,
                                        top, bottom,
                                        abyss);

  cl_cmatrix = gegl_clCreateBuffer (gegl_cl_get_context(),
                                    CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY,
                                    clen * sizeof(cl_float), cmatrix, &cl_err);
  CL_CHECK;

  while (gegl_buffer_cl_iterator_next (i, &err) && !err)
    {
      err = cl_gaussian_blur(i->tex[read],
                             i->tex[0],
                             &i->roi[0],
                             cl_cmatrix,
                             clen,
                             orientation);

      if (err)
        {
          gegl_buffer_cl_iterator_stop (i);
          break;
        }
    }

  cl_err = gegl_clReleaseMemObject (cl_cmatrix);
  CL_CHECK;

  cl_cmatrix = NULL;

  return !err;

error:
  if (cl_cmatrix)
    gegl_clReleaseMemObject (cl_cmatrix);

  return FALSE;
}

typedef struct
{
  GeglOrientation orientation;
  GeglRectangle   result;
  cl_mem          state;
  cl_float4       b_hi, b_lo, mb[3];
  cl_int          clamp_edges;
  cl_float4       edge;
  cl_int          n_comp;
} IirClParams;

/* Run one of the IIR kernels on every chunk of the line segment seg
 * (a part of p->result that is not split along the lines by the iterator).
 */
static gboolean
iir_cl_run_segment (const IirClParams   *p,
                    GeglBuffer          *src,
                    GeglBuffer          *dst,
                    const GeglRectangle *seg,
                    const Babl          *format,
                    gboolean             causal,
                    cl_int               first,
                    cl_int               last)
{
  gboolean              err = FALSE;
  cl_int                cl_err;
  GeglBufferClIterator *i;
  gint                  read;

  i = gegl_buffer_cl_iterator_new (dst, seg, format, GEGL_CL_BUFFER_WRITE);

  /* The CPU path reads the lines with GEGL_ABYSS_NONE and applies the
   * abyss policy only through iminus / uplus.
   */
  read = gegl_buffer_cl_iterator_add (i, src, seg, format,
                                      GEGL_CL_BUFFER_READ, GEGL_ABYSS_NONE);

  while (gegl_buffer_cl_iterator_next (i, &err) && !err)
    {
      const GeglRectangle *roi = &i->roi[0];
      cl_kernel            kernel;
      cl_int               len, n_lines, step, line_stride, line_offset;
      size_t               global_ws;

      if (p->orientation == GEGL_ORIENTATION_HORIZONTAL)
        {
          len         = roi->width;
          n_lines     = roi->height;
          step        = 1;
          line_stride = roi->width;
          line_offset = roi->y - p->result.y;
        }
      else
        {
          len         = roi->height;
          n_lines     = roi->width;
          step        = roi->width;
          line_stride = 1;
          line_offset = roi->x - p->result.x;
        }

      if (causal)
        {
          kernel = cl_data->kernel[2];
          cl_err = gegl_cl_set_kernel_args (kernel,
                                  sizeof(cl_mem),    (void*)&i->tex[read],
                                  sizeof(cl_mem),    (void*)&i->tex[0],
                                  sizeof(cl_mem),    (void*)&p->state,
                                  sizeof(cl_int),    (void*)&len,
                                  sizeof(cl_int),    (void*)&n_lines,
                                  sizeof(cl_int),    (void*)&step,
                                  sizeof(cl_int),    (void*)&line_stride,
                                  sizeof(cl_int),    (void*)&line_offset,
                                  sizeof(cl_int),    (void*)&p->n_comp,
                                  sizeof(cl_int),    (void*)&first,
                                  sizeof(cl_int),    (void*)&last,
                                  sizeof(cl_float4), (void*)&p->b_hi,
                                  sizeof(cl_float4), (void*)&p->b_lo,
                                  sizeof(cl_float4), (void*)&p->mb[0],
                                  sizeof(cl_float4), (void*)&p->mb[1],
                                  sizeof(cl_float4), (void*)&p->mb[2],
                                  sizeof(cl_int),    (void*)&p->clamp_edges,
                                  sizeof(cl_float4), (void*)&p->edge,
                                  NULL);
        }
      else
        {
          kernel = cl_data->kernel[3];
          cl_err = gegl_cl_set_kernel_args (kernel,
                                  sizeof(cl_mem),    (void*)&i->tex[read],
                                  sizeof(cl_mem),    (void*)&i->tex[0],
                                  sizeof(cl_mem),    (void*)&p->state,
                                  sizeof(cl_int),    (void*)&len,
                                  sizeof(cl_int),    (void*)&n_lines,
                                  sizeof(cl_int),    (void*)&step,
                                  sizeof(cl_int),    (void*)&line_stride,
                                  sizeof(cl_int),    (void*)&line_offset,
                                  sizeof(cl_int),    (void*)&p->n_comp,
                                  sizeof(cl_float4), (void*)&p->b_hi,
                                  sizeof(cl_float4), (void*)&p->b_lo,
                                  NULL);
        }
      CL_CHECK;

      global_ws = (size_t) n_lines * p->n_comp;

      cl_err = gegl_clEnqueueNDRangeKernel (gegl_cl_get_command_queue (),
                                            kernel, 1,
                                            NULL, &global_ws, NULL,
                                            0, NULL, NULL);
      CL_CHECK;

      /* single segment: the anti-causal pass runs in place on the
       * causal result of the same chunk
       */
      if (causal && first && last)
        {
          kernel = cl_data->kernel[3];
          cl_err = gegl_cl_set_kernel_args (kernel,
                                  sizeof(cl_mem),    (void*)&i->tex[0],
                                  sizeof(cl_mem),    (void*)&i->tex[0],
                                  sizeof(cl_mem),    (void*)&p->state,
                                  sizeof(cl_int),    (void*)&len,
                                  sizeof(cl_int),    (void*)&n_lines,
                                  sizeof(cl_int),    (void*)&step,
                                  sizeof(cl_int),    (void*)&line_stride,
                                  sizeof(cl_int),    (void*)&line_offset,
                                  sizeof(cl_int),    (void*)&p->n_comp,
                                  sizeof(cl_float4), (void*)&p->b_hi,
                                  sizeof(cl_float4), (void*)&p->b_lo,
                                  NULL);
          CL_CHECK;

          cl_err = gegl_clEnqueueNDRangeKernel (gegl_cl_get_command_queue (),
                                                kernel, 1,
                                                NULL, &global_ws, NULL,
                                                0, NULL, NULL);
          CL_CHECK;
        }
    }

  return !err;

error:
  gegl_buffer_cl_iterator_stop (i);
  return FALSE;
}

static gboolean
iir_cl_process (GeglBuffer            *input,
                GeglBuffer            *output,
                const GeglRectangle   *result,
                const Babl            *format,
                const gdouble         *b,
                gdouble              (*m)[3],
                GeglOrientation        orientation,
                GeglAbyssPolicy        abyss)
{
  IirClParams  p;
  gboolean     ok = TRUE;
  cl_int       cl_err;
  gdouble      bc[4], mb[3][3];
  gint         n_lines, len, seg_len, n_segs, s, k;
  GeglBuffer  *tmp = NULL;

  gblur_1d_cl_build ();

  if (!cl_data)
    return FALSE;

  /* gblur-1d works in float formats with 1 to 5 components; the
   * boundary tables only cover up to 4.
   */
  p.n_comp = babl_format_get_n_components (format);

  if (p.n_comp < 1 || p.n_comp > 4 ||
      babl_format_get_bytes_per_pixel (format) != p.n_comp * sizeof (gfloat))
    return FALSE;

  p.orientation = orientation;
  p.result      = *result;

  if (orientation == GEGL_ORIENTATION_HORIZONTAL)
    {
      len     = result->width;
      n_lines = result->height;
      seg_len = gegl_cl_get_iter_width ();
    }
  else
    {
      len     = result->height;
      n_lines = result->width;
      seg_len = gegl_cl_get_iter_height ();
    }

  if (len <= 0 || n_lines <= 0)
    return TRUE;

  n_segs = (len + seg_len - 1) / seg_len;

  /* Same boundary values as get_boundaries () */
  p.clamp_edges = 0;
  for (k = 0; k < 4; k++)
    p.edge.s[k] = 0.0f;

  switch (abyss)
    {
    case GEGL_ABYSS_CLAMP:
    default:
      p.clamp_edges = 1;
      break;
    case GEGL_ABYSS_NONE:
      break;
    case GEGL_ABYSS_WHITE:
      for (k = 0; k < p.n_comp; k++)
        p.edge.s[k] = white[k];
      break;
    case GEGL_ABYSS_BLACK:
      for (k = 0; k < p.n_comp; k++)
        p.edge.s[k] = black[(p.n_comp == 2 ? 2 : 0) + k];
      break;
    }

  /* See the comments in gblur-1d.cl */
  bc[0] = b[0];
  bc[1] = b[1];
  bc[2] = b[2] + 2.0 * b[3];
  bc[3] = b[3];

  for (k = 0; k < 4; k++)
    {
      p.b_hi.s[k] = bc[k];
      p.b_lo.s[k] = bc[k] - (gdouble) p.b_hi.s[k];
    }

  for (k = 0; k < 3; k++)
    {
      mb[k][0] = m[k][0] + m[k][1] + m[k][2];
      mb[k][1] = m[k][1] + 2.0 * m[k][2];
      mb[k][2] = m[k][2];
    }

  for (k = 0; k < 3; k++)
    {
      gint j;

      for (j = 0; j < 3; j++)
        p.mb[k].s[j] = k == 0 ? mb[0][j] : mb[k][j] - mb[0][j];

      p.mb[k].s[3] = 0.0f;
    }

  p.state = gegl_clCreateBuffer (gegl_cl_get_context (),
                                 CL_MEM_READ_WRITE,
                                 (size_t) n_lines * p.n_comp * 6 *
                                 sizeof (cl_float),
                                 NULL, &cl_err);
  if (cl_err != CL_SUCCESS)
    return FALSE;

  if (n_segs == 1)
    {
      ok = iir_cl_run_segment (&p, input, output, result, format,
                               TRUE, 1, 1);
    }
  else
    {
      /* Causal pass over the segments in order into tmp, then the
       * anti-causal pass in reverse order from tmp into output.
       */
      tmp = gegl_buffer_new (result, format);

      for (s = 0; ok && s < n_segs; s++)
        {
          GeglRectangle seg = *result;

          if (orientation == GEGL_ORIENTATION_HORIZONTAL)
            {
              seg.x     = result->x + s * seg_len;
              seg.width = MIN (seg_len, result->x + result->width - seg.x);
            }
          else
            {
              seg.y      = result->y + s * seg_len;
              seg.height = MIN (seg_len, result->y + result->height - seg.y);
            }

          ok = iir_cl_run_segment (&p, input, tmp, &seg, format,
                                   TRUE, s == 0, s == n_segs - 1);
        }

      for (s = n_segs - 1; ok && s >= 0; s--)
        {
          GeglRectangle seg = *result;

          if (orientation == GEGL_ORIENTATION_HORIZONTAL)
            {
              seg.x     = result->x + s * seg_len;
              seg.width = MIN (seg_len, result->x + result->width - seg.x);
            }
          else
            {
              seg.y      = result->y + s * seg_len;
              seg.height = MIN (seg_len, result->y + result->height - seg.y);
            }

          ok = iir_cl_run_segment (&p, tmp, output, &seg, format,
                                   FALSE, 0, 0);
        }

      g_object_unref (tmp);
    }

  gegl_clReleaseMemObject (p.state);

  return ok;
}

static gfloat
gaussian_func_1d (gfloat x,
                  gfloat sigma)
{
  return (1.0 / (sigma * sqrt (2.0 * G_PI))) *
          exp (-(x * x) / (2.0 * sigma * sigma));
}

static gint
fir_calc_convolve_matrix_length (gfloat sigma)
{
#if 1
  /* an arbitrary precision */
  gint clen = sigma > GEGL_FLOAT_EPSILON ? ceil (sigma * 6.5) : 1;
  clen = clen + ((clen + 1) % 2);
  return clen;
#else
  if (sigma > GEGL_FLOAT_EPSILON)
    {
      gint x = 0;
      while (gaussian_func_1d (x, sigma) > 1e-3)
        x++;
      return 2 * x + 1;
    }
  else return 1;
#endif
}

static gint
fir_gen_convolve_matrix (gfloat   sigma,
                         gfloat **cmatrix)
{
  gint    clen;
  gfloat *cmatrix_p;

  clen = fir_calc_convolve_matrix_length (sigma);

  *cmatrix  = gegl_malloc (sizeof (gfloat) * clen);
  cmatrix_p = *cmatrix;

  if (clen == 1)
    {
      cmatrix_p [0] = 1;
    }
  else
    {
      gint    i;
      gdouble sum = 0;
      gint    half_clen = clen / 2;

      for (i = 0; i < clen; i++)
        {
          cmatrix_p [i] = gaussian_func_1d (i - half_clen, sigma);
          sum += cmatrix_p [i];
        }

      for (i = 0; i < clen; i++)
        {
          cmatrix_p [i] /= sum;
        }
    }

  return clen;
}

static GeglGblur1dFilter
filter_disambiguation (GeglGblur1dFilter filter,
                       gfloat            std_dev)
{
  if (filter == GEGL_GBLUR_1D_AUTO)
    {
      /* Threshold 1.0 is arbitrary - but we really do not want IIR for much
         smaller stdevs */
      if (std_dev < 1.0)
        filter = GEGL_GBLUR_1D_FIR;
      else
        filter = GEGL_GBLUR_1D_IIR;
    }
  return filter;
}


/**********************************************
 *
 * GEGL Operation API
 *
 **********************************************/

static void
gegl_gblur_1d_prepare (GeglOperation *operation)
{
  const Babl *space = gegl_operation_get_source_space (operation, "input");
  GeglProperties *o = GEGL_PROPERTIES (operation);
  const Babl *src_format = gegl_operation_get_source_format (operation, "input");
  const char *format     = "RaGaBaA float";
  o->user_data = iir_young_blur_1D_rgbA;

  /*
   * FIXME: when the abyss policy is _NONE, the behavior at the edge
   *        depends on input format (with or without an alpha component)
   */
  if (src_format)
    {
      const Babl *model = babl_format_get_model (src_format);

      if (babl_model_is (model, "RGB") ||
          babl_model_is (model, "R'G'B'"))
        {
          format = "RGB float";
          o->user_data = iir_young_blur_1D_rgb;
        }
      else if (babl_model_is (model, "Y") || babl_model_is (model, "Y'"))
        {
          format = "Y float";
          o->user_data = iir_young_blur_1D_y;
        }
      else if (babl_model_is (model, "YA") || babl_model_is (model, "Y'A") ||
               babl_model_is (model, "YaA") || babl_model_is (model, "Y'aA"))
        {
          format = "YaA float";
          o->user_data = iir_young_blur_1D_yA;
        }
      else if (babl_model_is (model, "cmyk"))
        {
          format = "cmyk float";
          o->user_data = iir_young_blur_1D_generic;
        }
      else if (babl_model_is (model, "CMYK"))
        {
          format = "CMYK float";
          o->user_data = iir_young_blur_1D_generic;
        }
      else if (babl_model_is (model, "cmykA") ||
               babl_model_is (model, "camayakaA") ||
               babl_model_is (model, "CMYKA") ||
               babl_model_is (model, "CaMaYaKaA"))
        {
          format = "camayakaA float";
          o->user_data = iir_young_blur_1D_generic;
        }
    }

  gegl_operation_set_format (operation, "input", babl_format_with_space (format, space));
  gegl_operation_set_format (operation, "output", babl_format_with_space (format, space));
}

static GeglRectangle
gegl_gblur_1d_enlarge_extent (GeglProperties          *o,
                              const GeglRectangle *input_extent)
{
  gint clen = fir_calc_convolve_matrix_length (o->std_dev);

  GeglRectangle bounding_box = *input_extent;

  if (o->orientation == GEGL_ORIENTATION_HORIZONTAL)
    {
      bounding_box.x     -= clen / 2;
      bounding_box.width += clen - 1;
    }
  else
    {
      bounding_box.y      -= clen / 2;
      bounding_box.height += clen - 1;
    }

  return bounding_box;
}

static GeglRectangle
gegl_gblur_1d_get_required_for_output (GeglOperation       *operation,
                                       const gchar         *input_pad,
                                       const GeglRectangle *output_roi)
{
  GeglRectangle        required_for_output = { 0, };
  GeglProperties      *o       = GEGL_PROPERTIES (operation);
  GeglGblur1dFilter    filter  = filter_disambiguation (o->filter, o->std_dev);

  if (filter == GEGL_GBLUR_1D_IIR)
    {
      const GeglRectangle *in_rect =
        gegl_operation_source_get_bounding_box (operation, input_pad);

      if (in_rect)
        {
          if (!gegl_rectangle_is_infinite_plane (in_rect))
            {
              required_for_output = *output_roi;

              if (o->orientation == GEGL_ORIENTATION_HORIZONTAL)
                {
                  required_for_output.x     = in_rect->x;
                  required_for_output.width = in_rect->width;
                }
              else
                {
                  required_for_output.y      = in_rect->y;
                  required_for_output.height = in_rect->height;
                }

              if (!o->clip_extent)
                required_for_output =
                  gegl_gblur_1d_enlarge_extent (o, &required_for_output);
            }
          /* pass-through case */
          else
            return *output_roi;
        }
    }
  else
    {
      required_for_output = gegl_gblur_1d_enlarge_extent (o, output_roi);
    }

  return required_for_output;
}

static GeglRectangle
gegl_gblur_1d_get_bounding_box (GeglOperation *operation)
{
  GeglProperties          *o       = GEGL_PROPERTIES (operation);
  const GeglRectangle *in_rect =
    gegl_operation_source_get_bounding_box (operation, "input");

  if (! in_rect)
    return *GEGL_RECTANGLE (0, 0, 0, 0);

  if (gegl_rectangle_is_infinite_plane (in_rect))
    return *in_rect;

  if (o->clip_extent)
    {
      return *in_rect;
    }
  else
    {
      /* We use the FIR convolution length for both the FIR and the IIR case */
      return gegl_gblur_1d_enlarge_extent (o, in_rect);
    }
}

static GeglRectangle
gegl_gblur_1d_get_cached_region (GeglOperation       *operation,
                                 const GeglRectangle *output_roi)
{
  GeglRectangle      cached_region;
  GeglProperties    *o       = GEGL_PROPERTIES (operation);
  GeglGblur1dFilter  filter  = filter_disambiguation (o->filter, o->std_dev);

  cached_region = *output_roi;

  if (filter == GEGL_GBLUR_1D_IIR)
    {
      const GeglRectangle in_rect =
        gegl_gblur_1d_get_bounding_box (operation);

      if (! gegl_rectangle_is_empty (&in_rect) &&
          ! gegl_rectangle_is_infinite_plane (&in_rect))
        {
          GeglProperties *o = GEGL_PROPERTIES (operation);

          cached_region = *output_roi;

          if (o->orientation == GEGL_ORIENTATION_HORIZONTAL)
            {
              cached_region.x     = in_rect.x;
              cached_region.width = in_rect.width;
            }
          else
            {
              cached_region.y      = in_rect.y;
              cached_region.height = in_rect.height;
            }
        }
    }

  return cached_region;
}

static GeglSplitStrategy
gegl_gblur_1d_get_split_strategy (GeglOperation        *operation,
                                  GeglOperationContext *context,
                                  const gchar          *output_prop,
                                  const GeglRectangle  *result,
                                  gint                  level)
{
  GeglProperties *o = GEGL_PROPERTIES (operation);

  if (o->orientation == GEGL_ORIENTATION_HORIZONTAL)
    return GEGL_SPLIT_STRATEGY_HORIZONTAL;
  else
    return GEGL_SPLIT_STRATEGY_VERTICAL;
}

static GeglAbyssPolicy
to_gegl_policy (GeglGblur1dPolicy policy)
{
  switch (policy)
    {
    case (GEGL_GBLUR_1D_ABYSS_NONE):
      return GEGL_ABYSS_NONE;
      break;
    case (GEGL_GBLUR_1D_ABYSS_CLAMP):
      return GEGL_ABYSS_CLAMP;
      break;
    case (GEGL_GBLUR_1D_ABYSS_WHITE):
      return GEGL_ABYSS_WHITE;
      break;
    case (GEGL_GBLUR_1D_ABYSS_BLACK):
      return GEGL_ABYSS_BLACK;
      break;
    default:
      g_warning ("gblur-1d: unsupported abyss policy");
      return GEGL_ABYSS_NONE;
    }
}

static gboolean
gegl_gblur_1d_process (GeglOperation       *operation,
                       GeglBuffer          *input,
                       GeglBuffer          *output,
                       const GeglRectangle *result,
                       gint                 level)
{
  GeglProperties *o       = GEGL_PROPERTIES (operation);
  const Babl     *format  = gegl_operation_get_format (operation, "output");
  gfloat          std_dev = o->std_dev;

  GeglGblur1dFilter filter;
  GeglAbyssPolicy   abyss_policy = to_gegl_policy (o->abyss_policy);

  GeglRectangle rect2;
  if (level)
  {
    /*
      if a thread is asked to render rows from result->y to
      result->y + result->height at a level, it means that the thread
      rendering the chunk below will render starting from the row
      (result->y + result->height) >> level

      XXX: Probably it is also better to double check that it works seamlessly
      around the origin (I mean I don't remember whether shifting negative
      numbers rounds toward 0 or not)
    */
    rect2 = *result;
    rect2.x      = result->x >> level;
    rect2.y      = result->y >> level;
    rect2.width  = ((result->width + result->x) >> level) - rect2.x;
    rect2.height = ((result->height + result->y) >> level) - rect2.y;
    result = &rect2;
    std_dev = std_dev * (1.0/(1<<level));
  }
  filter = filter_disambiguation (o->filter, std_dev);

  if (filter == GEGL_GBLUR_1D_IIR)
    {
      IirYoungBlur1dFunc real_blur_1D = (IirYoungBlur1dFunc) o->user_data;
      gdouble b[4], m[3][3];

      iir_young_find_constants (std_dev, b, m);

      if (gegl_operation_use_opencl (operation) && level == 0)
        if (iir_cl_process (input, output, result, format,
                            b, m, o->orientation, abyss_policy))
          return TRUE;

      if (o->orientation == GEGL_ORIENTATION_HORIZONTAL)
        iir_young_hor_blur (real_blur_1D, input, result, output, b, m, abyss_policy, format, level);
      else
        iir_young_ver_blur (real_blur_1D, input, result, output, b, m, abyss_policy, format, level);
    }
  else
    {
      gfloat *cmatrix;
      gint    clen;

      clen = fir_gen_convolve_matrix (std_dev, &cmatrix);

      /* FIXME: implement others format cases */
      /* The OpenCL path only reads and writes level 0; at a mipmap
       * level it would blur the wrong pixels, so use the CPU path.
       */
      if (level == 0 &&
          gegl_operation_use_opencl (operation) &&
          format == babl_format ("RaGaBaA float"))
        if (fir_cl_process(input, output, result, format,
                           cmatrix, clen, o->orientation, abyss_policy))
        {
          gegl_free (cmatrix);
          return TRUE;
        }

      if (o->orientation == GEGL_ORIENTATION_HORIZONTAL)
        fir_hor_blur (input, result, output, cmatrix, clen, abyss_policy, format, level);
      else
        fir_ver_blur (input, result, output, cmatrix, clen, abyss_policy, format, level);

      gegl_free (cmatrix);
    }

  return  TRUE;
}

/* Pass-through when trying to perform IIR on an infinite plane
 */
static gboolean
operation_process (GeglOperation        *operation,
                   GeglOperationContext *context,
                   const gchar          *output_prop,
                   const GeglRectangle  *result,
                   gint                  level)
{
  GeglOperationClass  *operation_class;
  GeglProperties          *o = GEGL_PROPERTIES (operation);
  GeglGblur1dFilter    filter  = filter_disambiguation (o->filter, o->std_dev);

  operation_class = GEGL_OPERATION_CLASS (gegl_op_parent_class);

  if (filter == GEGL_GBLUR_1D_IIR)
    {
      const GeglRectangle *in_rect =
        gegl_operation_source_get_bounding_box (operation, "input");

      if (in_rect && gegl_rectangle_is_infinite_plane (in_rect))
        {
          gpointer in = gegl_operation_context_get_object (context, "input");
          gegl_operation_context_take_object (context, "output",
                                              g_object_ref (G_OBJECT (in)));
          return TRUE;
        }
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

  filter_class->process                    = gegl_gblur_1d_process;
  filter_class->get_split_strategy         = gegl_gblur_1d_get_split_strategy;
  operation_class->prepare                 = gegl_gblur_1d_prepare;
  operation_class->process                 = operation_process;
  operation_class->get_bounding_box        = gegl_gblur_1d_get_bounding_box;
  operation_class->get_required_for_output = gegl_gblur_1d_get_required_for_output;
  operation_class->get_cached_region       = gegl_gblur_1d_get_cached_region;
  operation_class->opencl_support          = TRUE;
  gegl_operation_class_set_keys (operation_class,
    "name",           "gegl:gblur-1d",
    "categories",     "hidden:blur",
    "title",          _("1D Gaussian-blur"),
    "reference-hash", "559224424d47c48596ea331b3d4f4a5a",
    "description",
        _("Performs an averaging of neighboring pixels with the "
          "normal distribution as weighting"),
        NULL);
}

#endif
