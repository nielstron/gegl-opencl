/* SPDX-License-Identifier: LGPL-3.0-or-later */

#ifndef VARIABLE_BLUR_PARAMS_H
#define VARIABLE_BLUR_PARAMS_H

#include <glib.h>
#include <math.h>

#define VARIABLE_BLUR_MAX_LEVELS 16
#define VARIABLE_BLUR_GAMMA      1.5

static inline gint
variable_blur_levels (gdouble  radius,
                      gboolean high_quality)
{
  return high_quality ? VARIABLE_BLUR_MAX_LEVELS :
    ceil (CLAMP (log (radius) / G_LN2 + 3, 2, VARIABLE_BLUR_MAX_LEVELS));
}

static inline gdouble
variable_blur_sigma (gdouble radius,
                     gint    level,
                     gint    levels,
                     gdouble gamma)
{
  return radius * pow ((gdouble) level / (levels - 1), gamma);
}

#endif
