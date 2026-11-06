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
 */

float
randomize_value (__global const int        *random_data,
                          const GeglRandom  rand,
                                float       now,
                                float       min_val,
                                float       max_val,
                                bool        wraps_around,
                                float       rand_max,
                                int         holdness,
                                int         x,
                                int         y,
                                int         n)
{
  const float steps = max_val - min_val;
  float rand_val = gegl_cl_random_float (random_data, rand, x, y, 0, n++);

  for (int i = 1; i < holdness; i++)
    {
      float tmp = gegl_cl_random_float (random_data, rand, x, y, 0, n++);
      rand_val = min (rand_val, tmp);
    }

  const int flag = (gegl_cl_random_float (random_data, rand, x, y, 0, n) < 0.5)
                   ? -1 : 1;
  float new_val = now + flag * fmod (rand_max * rand_val, steps);

  if (new_val < min_val)
    {
      if (wraps_around)
        new_val += steps;
      else
        new_val = min_val;
    }

  if (max_val < new_val)
    {
      if (wraps_around)
        new_val -= steps;
      else
        new_val = max_val;
    }

  return new_val;
}

/* Same as babl's rgba_to_hsv_step (extensions/HSV.c), applied to
 * R'G'B' values (babl's HSV is defined on sRGB-TRC encoded values).
 */
float4
rgb_to_hsv (float4 rgba)
{
  const float epsilon = 1.0e-10f;
  float red   = rgba.x;
  float green = rgba.y;
  float blue  = rgba.z;
  float hue, saturation, value, minv, chroma;

  if (red > green)
    {
      value = (red < blue) ? blue : red;
      minv  = (green > blue) ? blue : green;
    }
  else
    {
      value = (green < blue) ? blue : green;
      minv  = (red > blue) ? blue : red;
    }

  chroma = value - minv;

  if (value < epsilon)
    saturation = 0.0f;
  else
    saturation = chroma / value;

  if (saturation < epsilon)
    {
      hue = 0.0f;
    }
  else
    {
      if (fabs (red - value) < epsilon)
        {
          hue = (green - blue) / chroma;

          if (hue < 0.0f)
            hue += 6.0f;
        }
      else if (fabs (green - value) < epsilon)
        hue = 2.0f + (blue - red) / chroma;
      else
        hue = 4.0f + (red - green) / chroma;

      hue /= 6.0f;
    }

  return (float4)(hue, saturation, value, rgba.w);
}

/* Same as babl's hsv_to_rgba_step, producing R'G'B' values. */
float4
hsv_to_rgb (float4 hsva)
{
  const float chroma = hsva.y * hsva.z;
  float red = 0.0f, green = 0.0f, blue = 0.0f;
  float h_tmp, x, minv;

  h_tmp  = fmod (hsva.x, 1.0f);
  h_tmp += (h_tmp < 0.0f) ? 1.0f : 0.0f;
  h_tmp *= 6.0f;

  x = chroma * (1.0f - fabs (fmod (h_tmp, 2.0f) - 1.0f));

  if (h_tmp < 1.0f)
    {
      red   = chroma;
      green = x;
    }
  else if (h_tmp < 2.0f)
    {
      red   = x;
      green = chroma;
    }
  else if (h_tmp < 3.0f)
    {
      green = chroma;
      blue  = x;
    }
  else if (h_tmp < 4.0f)
    {
      green = x;
      blue  = chroma;
    }
  else if (h_tmp < 5.0f)
    {
      red  = x;
      blue = chroma;
    }
  else if (h_tmp < 6.0f)
    {
      red  = chroma;
      blue = x;
    }

  minv = hsva.z - chroma;

  return (float4)(red + minv, green + minv, blue + minv, hsva.w);
}

float4
noise_hsv (float4                    in_v,
           __global const int       *random_data,
           const GeglRandom          rand,
           int                       x,
           int                       y,
           int                       whole_region_width,
           int                       holdness,
           float                     hue_distance,
           float                     saturation_distance,
           float                     value_distance)
{
  int n = (3 * holdness + 4) * (x + whole_region_width * y);

  float hue        = in_v.s0;
  float saturation = in_v.s1;
  float value      = in_v.s2;
  float alpha      = in_v.s3;

  /* there is no need for scattering hue of desaturated pixels here */
  if ((hue_distance > 0) && (saturation > 0))
    hue = randomize_value (random_data, rand,
                           hue, 0.0, 1.0, true,
                           hue_distance, holdness,
                           x, y, n);

  n += holdness + 1;
  if (saturation_distance > 0)
    {
      /* desaturated pixels get random hue before increasing saturation */
      if (saturation == 0)
        hue = gegl_cl_random_float_range (random_data, rand,
                                          x, y, 0, n, 0.0, 1.0);
      saturation = randomize_value (random_data, rand,
                                    saturation, 0.0, 1.0, false,
                                    saturation_distance, holdness,
                                    x, y, n + 1);
    }

  n += holdness + 2;
  if (value_distance > 0)
    value = randomize_value (random_data, rand,
                             value, 0.0, 1.0, false,
                             value_distance, holdness,
                             x, y, n);

  return (float4)(hue, saturation, value, alpha);
}

/* HSVA float in and out */
__kernel void cl_noise_hsv(__global const float4    *in,
                           __global       float4    *out,
                           __global const int       *random_data,
                                    const GeglRandom rand,
                                          int        x_offset,
                                          int        y_offset,
                                          int        roi_width,
                                          int        whole_region_width,
                                          int        holdness,
                                          float      hue_distance,
                                          float      saturation_distance,
                                          float      value_distance)
{
  const int gid  = get_global_id(0);
  const int gidy = gid / roi_width;
  const int gidx = gid - gidy * roi_width;

  out[gid] = noise_hsv (in[gid], random_data, rand,
                        gidx + x_offset, gidy + y_offset,
                        whole_region_width, holdness,
                        hue_distance, saturation_distance, value_distance);
}

/* R'G'B'A float in and out, converting to and from HSV in the kernel, so
 * that the buffer conversions can stay on the GPU.
 */
__kernel void cl_noise_hsv_rgb(__global const float4    *in,
                               __global       float4    *out,
                               __global const int       *random_data,
                                        const GeglRandom rand,
                                              int        x_offset,
                                              int        y_offset,
                                              int        roi_width,
                                              int        whole_region_width,
                                              int        holdness,
                                              float      hue_distance,
                                              float      saturation_distance,
                                              float      value_distance)
{
  const int gid  = get_global_id(0);
  const int gidy = gid / roi_width;
  const int gidx = gid - gidy * roi_width;

  const float4 hsva = noise_hsv (rgb_to_hsv (in[gid]), random_data, rand,
                                 gidx + x_offset, gidy + y_offset,
                                 whole_region_width, holdness,
                                 hue_distance, saturation_distance,
                                 value_distance);

  out[gid] = hsv_to_rgb (hsva);
}
