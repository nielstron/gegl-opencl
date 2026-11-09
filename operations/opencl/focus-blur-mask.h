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
 */

#ifndef FOCUS_BLUR_MASK_H
#define FOCUS_BLUR_MASK_H

#include <math.h>

#include "opencl/gegl-cl.h"
#include "opencl/vignette.cl.h"

typedef struct
{
  GeglRectangle extent;
  cl_int        midx;
  cl_int        midy;
  cl_int        shape;
  cl_float      scale;
  cl_float      cost;
  cl_float      sint;
  cl_float      gamma;
  cl_float      length;
  cl_float      radius0;
  cl_float      rdiff;
  cl_float4     luminance;
} FocusBlurMask;

/* Match vignette.c with proportion = 0 on an opaque black input. */
static inline gboolean
focus_blur_mask_init (FocusBlurMask       *mask,
                      const GeglRectangle *extent,
                      gint                 shape,
                      gdouble              radius,
                      gdouble              softness,
                      gdouble              gamma,
                      gdouble              squeeze,
                      gdouble              x,
                      gdouble              y,
                      gdouble              rotation)
{
  gint64  right  = (gint64) extent->x + extent->width;
  gint64  bottom = (gint64) extent->y + extent->height;
  gdouble midx   = extent->x + extent->width * x;
  gdouble midy   = extent->y + extent->height * y;
  gfloat  aspect = squeeze;
  gfloat  radius1;
  gdouble luminance[3];

  /* Vignette uses signed integer pixel coordinates and offsets. */
  if (extent->width <= 0 || extent->height <= 0 ||
      right > G_MAXINT || bottom > G_MAXINT ||
      !isfinite (midx) || midx < G_MININT || midx > G_MAXINT ||
      !isfinite (midy) || midy < G_MININT || midy > G_MAXINT ||
      !isfinite (radius) || radius > G_MAXFLOAT)
    return FALSE;

  mask->extent = *extent;
  mask->midx   = midx;
  mask->midy   = midy;
  mask->shape  = shape;

  if ((gint64) extent->x - mask->midx < G_MININT ||
      (gint64) extent->y - mask->midy < G_MININT ||
      right  - 1 - mask->midx > G_MAXINT ||
      bottom - 1 - mask->midy > G_MAXINT)
    return FALSE;

  mask->scale = aspect < 0.0f ? 1.0 / (tan (-aspect * (G_PI / 2)) + 1.0)
                              : tan (aspect * (G_PI / 2)) + 1.0;
  mask->length = extent->width / 2.0;
  if (mask->scale > 1.0f)
    mask->length /= mask->scale;

  mask->radius0 = radius * (1.0 - softness);
  radius1 = radius;
  mask->rdiff = radius1 - mask->radius0;
  if (fabs (mask->rdiff) < 0.0001)
    mask->rdiff = 0.0001;

  mask->gamma = MAX (gamma, 0.0001);
  if (mask->gamma > 1.9999 && mask->gamma < 2.0001)
    mask->gamma = 2.0f;

  mask->cost = cos (-rotation * (G_PI * 2 / 360.0));
  mask->sint = sin (-rotation * (G_PI * 2 / 360.0));

  babl_space_get_rgb_luminance (babl_format_get_space (babl_format ("RGBA float")),
                               &luminance[0], &luminance[1], &luminance[2]);
  mask->luminance.s[0] = luminance[0];
  mask->luminance.s[1] = luminance[1];
  mask->luminance.s[2] = luminance[2];
  mask->luminance.s[3] = 0.0f;

  return TRUE;
}

/* Write a packed Y float mask for rect, including zero-valued abyss pixels. */
static inline gboolean
focus_blur_mask_enqueue (const FocusBlurMask  *mask,
                         const GeglRectangle *rect,
                         cl_mem               output)
{
  const char    *kernel_names[] = { "vignette_cl", NULL };
  const size_t   global_size[] = { rect->width, rect->height };
  cl_int         right  = mask->extent.x + mask->extent.width;
  cl_int         bottom = mask->extent.y + mask->extent.height;
  GeglClRunData *data;
  cl_kernel      kernel;
  cl_int         cl_err;

  if (rect->width <= 0 || rect->height <= 0 ||
      (gint64) rect->x + rect->width  - 1 > G_MAXINT ||
      (gint64) rect->y + rect->height - 1 > G_MAXINT)
    return FALSE;

  data = gegl_cl_compile_and_build (vignette_cl_source, kernel_names);
  if (!data)
    return FALSE;

  /* Kernel arguments belong to a single render, even with a cached program. */
  kernel = gegl_clCreateKernel (data->program, "focus_blur_mask", &cl_err);
  CL_CHECK;

  cl_err = gegl_cl_set_kernel_args (kernel,
                                   sizeof (cl_mem),    &output,
                                   sizeof (cl_int),    &rect->x,
                                   sizeof (cl_int),    &rect->y,
                                   sizeof (cl_int),    &mask->midx,
                                   sizeof (cl_int),    &mask->midy,
                                   sizeof (cl_int),    &mask->shape,
                                   sizeof (cl_float),  &mask->scale,
                                   sizeof (cl_float),  &mask->cost,
                                   sizeof (cl_float),  &mask->sint,
                                   sizeof (cl_float),  &mask->gamma,
                                   sizeof (cl_float),  &mask->length,
                                   sizeof (cl_float),  &mask->radius0,
                                   sizeof (cl_float),  &mask->rdiff,
                                   sizeof (cl_int),    &mask->extent.x,
                                   sizeof (cl_int),    &mask->extent.y,
                                   sizeof (cl_int),    &right,
                                   sizeof (cl_int),    &bottom,
                                   sizeof (cl_float4), &mask->luminance,
                                   NULL);
  CL_CHECK;

  cl_err = gegl_clEnqueueNDRangeKernel (gegl_cl_get_command_queue (), kernel, 2,
                                       NULL, global_size, NULL, 0, NULL, NULL);
  CL_CHECK;

  gegl_clReleaseKernel (kernel);
  return TRUE;

error:
  if (kernel)
    gegl_clReleaseKernel (kernel);
  return FALSE;
}

#endif
