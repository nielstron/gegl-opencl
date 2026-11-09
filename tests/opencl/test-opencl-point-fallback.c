/* SPDX-License-Identifier: GPL-3.0-or-later */

#include <string.h>

#include "gegl.h"
#include "opencl/gegl-cl.h"

static clEnqueueNDRangeKernel_fn enqueue;
static guint                    dispatches;

static cl_int CL_API_CALL
fail_second_invert (cl_command_queue     queue,
                    cl_kernel            kernel,
                    cl_uint              dimensions,
                    const size_t        *offset,
                    const size_t        *global,
                    const size_t        *local,
                    cl_uint              wait_count,
                    const cl_event      *wait_list,
                    cl_event            *event)
{
  gchar name[128] = { 0 };

  gegl_clGetKernelInfo (kernel, CL_KERNEL_FUNCTION_NAME,
                        sizeof (name), name, NULL);
  if (strcmp (name, "gegl_invert_linear") == 0 && ++dispatches == 2)
    return CL_INVALID_OPERATION;

  return enqueue (queue, kernel, dimensions, offset, global, local,
                  wait_count, wait_list, event);
}

static void
render (GeglBuffer          *input,
        const GeglRectangle *roi,
        const Babl          *format,
        gboolean             use_opencl,
        gfloat              *pixels)
{
  GeglNode *graph = gegl_node_new ();
  GeglNode *source, *before, *invert, *after;

  g_object_set (graph, "use-opencl", use_opencl, NULL);
  source = gegl_node_new_child (graph, "operation", "gegl:buffer-source",
                                "buffer", input, NULL);
  before = gegl_node_new_child (graph, "operation", "gegl:brightness-contrast", NULL);
  invert = gegl_node_new_child (graph, "operation", "gegl:invert-linear", NULL);
  after = gegl_node_new_child (graph, "operation", "gegl:brightness-contrast", NULL);
  gegl_node_link_many (source, before, invert, after, NULL);
  gegl_node_blit (after, 1.0, roi, format, pixels,
                  GEGL_AUTO_ROWSTRIDE, GEGL_BLIT_DEFAULT);
  g_object_unref (graph);
}

int
main (int    argc,
      char **argv)
{
  g_setenv ("GEGL_USE_OPENCL", "gpu", TRUE);
  gegl_init (&argc, &argv);

  if (! gegl_cl_is_accelerated ())
    {
      gegl_exit ();
      return 77;
    }

  const GeglRectangle roi = { 0, 0, 2 * gegl_cl_get_iter_width () + 1, 67 };
  const Babl *format = babl_format ("RGBA float");
  const gsize values = (gsize) roi.width * roi.height * 4;
  gfloat *source = g_new (gfloat, values);
  gfloat *reference = g_new0 (gfloat, values);
  gfloat *result = g_new0 (gfloat, values);
  GeglBuffer *input = gegl_buffer_new (&roi, format);

  for (gsize i = 0; i < values; i++)
    source[i] = (i % 7 + 1) / 8.0f;
  gegl_buffer_set (input, &roi, 0, format, source, GEGL_AUTO_ROWSTRIDE);
  render (input, &roi, format, FALSE, reference);

  enqueue = gegl_clEnqueueNDRangeKernel;
  gegl_clEnqueueNDRangeKernel = fail_second_invert;
  render (input, &roi, format, TRUE, result);
  gegl_clEnqueueNDRangeKernel = enqueue;

  const gboolean passed = dispatches == 2 &&
                         memcmp (reference, result, values * sizeof (gfloat)) == 0;
  if (! passed)
    g_printerr ("Unexpected point-filter output after %u OpenCL dispatches\n",
                dispatches);

  g_object_unref (input);
  g_free (source);
  g_free (reference);
  g_free (result);
  gegl_exit ();
  return passed ? 0 : 1;
}
