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
 *           2013 Daniel Sabo
 */

/* OpenCL Initialization
   The API is stubbed out so we detect if OpenCL libraries are available
   in runtime.
*/

#include "config.h"

#define __GEGL_CL_INIT_MAIN__
#include "gegl-cl-init.h"
#undef __GEGL_CL_INIT_MAIN__

#include <glib.h>
#include <gmodule.h>
#include <string.h>
#ifdef _WIN32
#define strdup _strdup
#endif
#ifdef __APPLE__
#include <dlfcn.h>
#endif
#include <stdio.h>

#include "gegl/gegl-debug.h"
#include "gegl/buffer/gegl-buffer.h"
#include "gegl/buffer/gegl-buffer-private.h"
#include "operations/opencl/random.cl.h"
#include "gegl-buffer-cl-cache.h"
#include "gegl-cl-color.h"
#include "gegl-cl.h"

const char *gegl_cl_errstring(cl_int err) {
  static const char* strings[] =
  {
    /* Error Codes */
      "success"                         /*  0  */
    , "device not found"                /* -1  */
    , "device not available"            /* -2  */
    , "compiler not available"          /* -3  */
    , "mem object allocation failure"   /* -4  */
    , "out of resources"                /* -5  */
    , "out of host memory"              /* -6  */
    , "profiling info not available"    /* -7  */
    , "mem copy overlap"                /* -8  */
    , "image format mismatch"           /* -9  */
    , "image format not supported"      /* -10 */
    , "build program failure"           /* -11 */
    , "map failure"                     /* -12 */
    , ""                                /* -13 */
    , ""                                /* -14 */
    , ""                                /* -15 */
    , ""                                /* -16 */
    , ""                                /* -17 */
    , ""                                /* -18 */
    , ""                                /* -19 */
    , ""                                /* -20 */
    , ""                                /* -21 */
    , ""                                /* -22 */
    , ""                                /* -23 */
    , ""                                /* -24 */
    , ""                                /* -25 */
    , ""                                /* -26 */
    , ""                                /* -27 */
    , ""                                /* -28 */
    , ""                                /* -29 */
    , "invalid value"                   /* -30 */
    , "invalid device type"             /* -31 */
    , "invalid platform"                /* -32 */
    , "invalid device"                  /* -33 */
    , "invalid context"                 /* -34 */
    , "invalid queue properties"        /* -35 */
    , "invalid command queue"           /* -36 */
    , "invalid host ptr"                /* -37 */
    , "invalid mem object"              /* -38 */
    , "invalid image format descriptor" /* -39 */
    , "invalid image size"              /* -40 */
    , "invalid sampler"                 /* -41 */
    , "invalid binary"                  /* -42 */
    , "invalid build options"           /* -43 */
    , "invalid program"                 /* -44 */
    , "invalid program executable"      /* -45 */
    , "invalid kernel name"             /* -46 */
    , "invalid kernel definition"       /* -47 */
    , "invalid kernel"                  /* -48 */
    , "invalid arg index"               /* -49 */
    , "invalid arg value"               /* -50 */
    , "invalid arg size"                /* -51 */
    , "invalid kernel args"             /* -52 */
    , "invalid work dimension"          /* -53 */
    , "invalid work group size"         /* -54 */
    , "invalid work item size"          /* -55 */
    , "invalid global offset"           /* -56 */
    , "invalid event wait list"         /* -57 */
    , "invalid event"                   /* -58 */
    , "invalid operation"               /* -59 */
    , "invalid gl object"               /* -60 */
    , "invalid buffer size"             /* -61 */
    , "invalid mip level"               /* -62 */
    , "invalid global work size"        /* -63 */
  };

  static const int strings_len = sizeof(strings) / sizeof(strings[0]);

  if (-err < 0 || -err >= strings_len)
    return "unknown error";

  return strings[-err];
}

static gboolean _gegl_cl_is_accelerated;

gboolean
gegl_cl_is_accelerated (void)
{
  return _gegl_cl_is_accelerated;
}

typedef struct
{
  gboolean            is_loaded;
  gboolean            have_opengl;
  gboolean            hard_disable;
  gboolean            enable_profiling;
  gboolean            platform_v3;
  gboolean            compiler_v3;
  cl_context          ctx;
  cl_platform_id      platform;
  cl_device_id        device;
  cl_command_queue    cq;
  cl_bool             image_support;
  size_t              iter_height;
  size_t              iter_width;
  cl_ulong            max_mem_alloc;
  cl_ulong            local_mem_size;
  cl_device_fp_config fp_config;

  char platform_name   [250];
  char platform_version[250];
  char platform_ext    [1024];
  char device_name     [250];
  char device_version  [250];
}
GeglClState;

/* we made some performance measurements and OpenCL in the CPU is rarely worth it,
 * specially now that we got our multi-threading working */

static cl_device_type gegl_cl_default_device_type = CL_DEVICE_TYPE_GPU;
static GeglClState cl_state = { 0, };
static GHashTable *cl_program_hash[2];
static GMutex cl_program_mutex;


gboolean
gegl_cl_has_gl_sharing (void)
{
  return cl_state.have_opengl && gegl_cl_is_accelerated ();
}

void
gegl_cl_disable (void)
{
  _gegl_cl_is_accelerated = FALSE;

  gegl_buffer_ext_flush = NULL;
  gegl_buffer_ext_invalidate = NULL;
  gegl_tile_handler_cache_ext_flush = NULL;
}

void
gegl_cl_hard_disable (void)
{
  cl_state.hard_disable = TRUE;
  _gegl_cl_is_accelerated = FALSE;

  gegl_buffer_ext_flush = NULL;
  gegl_buffer_ext_invalidate = NULL;
  gegl_tile_handler_cache_ext_flush = NULL;
}

cl_platform_id
gegl_cl_get_platform (void)
{
  return cl_state.platform;
}

cl_device_id
gegl_cl_get_device (void)
{
  return cl_state.device;
}

cl_context
gegl_cl_get_context (void)
{
  return cl_state.ctx;
}

cl_command_queue
gegl_cl_get_command_queue (void)
{
  return cl_state.cq;
}

cl_ulong
gegl_cl_get_local_mem_size (void)
{
  return cl_state.local_mem_size;
}

size_t
gegl_cl_get_iter_width (void)
{
  return cl_state.iter_width;
}

size_t
gegl_cl_get_iter_height (void)
{
  return cl_state.iter_height;
}

void
gegl_cl_set_profiling (gboolean enable)
{
  g_return_if_fail (!cl_state.is_loaded);

  cl_state.enable_profiling = enable;
}

void
gegl_cl_set_default_device_type (cl_device_type default_device_type)
{
  g_return_if_fail (!cl_state.is_loaded);

  gegl_cl_default_device_type = default_device_type;
}

static gboolean
gegl_cl_device_has_extension (cl_device_id device, const char *extension_name)
{
  cl_int     cl_err;
  size_t     string_len = 0;
  gchar     *device_ext_string = NULL;
  gchar    **extensions;
  gboolean   found = FALSE;

  if (!extension_name)
    return FALSE;

  cl_err= gegl_clGetDeviceInfo (device, CL_DEVICE_EXTENSIONS,
                                0, NULL, &string_len);
  CL_CHECK_ONLY (cl_err);

  if (!string_len)
    return FALSE;

  device_ext_string = g_malloc0 (string_len);

  cl_err = gegl_clGetDeviceInfo (device, CL_DEVICE_EXTENSIONS,
                                 string_len, device_ext_string, NULL);
  CL_CHECK_ONLY (cl_err);

  extensions = g_strsplit (device_ext_string, " ", 0);

  for (gint i = 0; extensions[i] && !found; ++i)
    {
      if (!strcmp (extensions[i], extension_name))
        found = TRUE;
    }

  g_free (device_ext_string);
  g_strfreev (extensions);

  return found;
}

gboolean
gegl_cl_has_extension (const char *extension_name)
{
  if (!gegl_cl_is_accelerated () || !extension_name)
    return FALSE;

  return gegl_cl_device_has_extension (cl_state.device, extension_name);
}

#if defined(__APPLE__)
#define GL_LIBRARY_NAME "/System/Library/Frameworks/OpenGL.framework/Versions/Current/OpenGL"
#define CL_LIBRARY_NAME "/System/Library/Frameworks/OpenCL.framework/Versions/Current/OpenCL"
#elif defined(G_OS_WIN32)
#define CL_LIBRARY_NAME "OpenCL.dll"
#elif defined(__ANDROID__)
#define CL_LIBRARY_NAME "libOpenCL.so"
/* Pixel devices ship the vendor library under a different name */
#define CL_LIBRARY_NAME_FALLBACK "libOpenCL-pixel.so"
#else
#define GL_LIBRARY_NAME "libGL.so.1"
#define CL_LIBRARY_NAME "libOpenCL.so.1"
#endif

#ifndef __APPLE__
#define CL_LOAD_FUNCTION(func)                                                    \
G_STMT_START                                                                      \
  {                                                                               \
    if (!g_module_symbol (module, #func, (gpointer *)&gegl_##func))               \
      {                                                                           \
        GEGL_NOTE (GEGL_DEBUG_OPENCL, "%s: %s", CL_LIBRARY_NAME,                  \
                   g_module_error ());                                            \
        if (!g_module_close (module))                                             \
          g_warning ("%s: %s", CL_LIBRARY_NAME, g_module_error ());               \
                                                                                  \
        return FALSE;                                                             \
      }                                                                           \
                                                                                  \
    if (gegl_##func == NULL)                                                      \
      {                                                                           \
        GEGL_NOTE (GEGL_DEBUG_OPENCL, "symbol gegl_##func is NULL");              \
        if (!g_module_close (module))                                             \
          g_warning ("%s: %s", CL_LIBRARY_NAME, g_module_error ());               \
                                                                                  \
        return FALSE;                                                             \
      }                                                                           \
  }                                                                               \
G_STMT_END
#else
#define CL_LOAD_FUNCTION(func)                                                    \
G_STMT_START                                                                      \
  {                                                                               \
    gegl_##func = dlsym (module, #func);                                          \
    if (gegl_##func == NULL)                                                      \
      {                                                                           \
        GEGL_NOTE (GEGL_DEBUG_OPENCL, "symbol gegl_##func is NULL");              \
        return FALSE;                                                             \
      }                                                                           \
  }                                                                               \
G_STMT_END
#endif

#define CL_LOAD_EXTENSION_FUNCTION(func)                                          \
G_STMT_START                                                                      \
  {                                                                               \
    g_assert (gegl_clGetExtensionFunctionAddressForPlatform != NULL);             \
                                                                                  \
    gegl_##func = gegl_clGetExtensionFunctionAddressForPlatform (platform,        \
                                                                 #func);          \
    if (gegl_##func == NULL)                                                      \
      {                                                                           \
        GEGL_NOTE (GEGL_DEBUG_OPENCL, "symbol gegl_##func is NULL");              \
        return FALSE;                                                             \
      }                                                                           \
  }                                                                               \
G_STMT_END

static gboolean
gegl_cl_init_common (cl_device_type requested_device_type);

gboolean
gegl_cl_init (void)
{
  return gegl_cl_init_common (gegl_cl_default_device_type);
}

static gboolean
gegl_cl_init_load_functions (void)
{
#ifndef __APPLE__
  GModule *module = g_module_open (CL_LIBRARY_NAME, G_MODULE_BIND_LAZY);
#ifdef CL_LIBRARY_NAME_FALLBACK
  if (!module)
    module = g_module_open (CL_LIBRARY_NAME_FALLBACK, G_MODULE_BIND_LAZY);
#endif
#else
  void *module = dlopen(CL_LIBRARY_NAME, RTLD_LAZY | RTLD_LOCAL);
#endif
  if (!module)
    {
      GEGL_NOTE (GEGL_DEBUG_OPENCL, "Unable to load OpenCL library %s", CL_LIBRARY_NAME);
      return FALSE;
    }

  /* Query Platform Info */
  CL_LOAD_FUNCTION (clGetPlatformIDs);
  CL_LOAD_FUNCTION (clGetPlatformInfo);

  /* Query Devices */
  CL_LOAD_FUNCTION (clGetDeviceIDs);
  CL_LOAD_FUNCTION (clGetDeviceInfo);

  /* Partition a Device */
  CL_LOAD_FUNCTION (clCreateSubDevices);
  CL_LOAD_FUNCTION (clReleaseDevice);
  CL_LOAD_FUNCTION (clRetainDevice);

  /* Contexts */
  CL_LOAD_FUNCTION (clCreateContext);
  CL_LOAD_FUNCTION (clCreateContextFromType);
  CL_LOAD_FUNCTION (clGetContextInfo);
  CL_LOAD_FUNCTION (clReleaseContext);
  CL_LOAD_FUNCTION (clRetainContext);
#ifndef __APPLE__
  CL_LOAD_FUNCTION (clSetContextDestructorCallback);
#endif

  /* Command-Queues */
#ifndef __APPLE__
  CL_LOAD_FUNCTION (clCreateCommandQueueWithProperties);
#else
  CL_LOAD_FUNCTION (clCreateCommandQueue);
#endif
  CL_LOAD_FUNCTION (clGetCommandQueueInfo);
  CL_LOAD_FUNCTION (clReleaseCommandQueue);
  CL_LOAD_FUNCTION (clRetainCommandQueue);
#ifndef __APPLE__
  CL_LOAD_FUNCTION (clSetDefaultDeviceCommandQueue);
#endif

  /* Buffer Objects */
  CL_LOAD_FUNCTION (clCreateBuffer);
#ifndef __APPLE__
  CL_LOAD_FUNCTION (clCreateBufferWithProperties);
#endif
  CL_LOAD_FUNCTION (clCreateSubBuffer);
  CL_LOAD_FUNCTION (clEnqueueReadBuffer);
  CL_LOAD_FUNCTION (clEnqueueWriteBuffer);
  CL_LOAD_FUNCTION (clEnqueueReadBufferRect);
  CL_LOAD_FUNCTION (clEnqueueWriteBufferRect);
  CL_LOAD_FUNCTION (clEnqueueCopyBuffer);
  CL_LOAD_FUNCTION (clEnqueueCopyBufferRect);
  CL_LOAD_FUNCTION (clEnqueueFillBuffer);
  CL_LOAD_FUNCTION (clEnqueueMapBuffer);

  /* Image Objects */
  CL_LOAD_FUNCTION (clCreateImage);
#ifndef __APPLE__
  CL_LOAD_FUNCTION (clCreateImageWithProperties);
#endif
  CL_LOAD_FUNCTION (clEnqueueReadImage);
  CL_LOAD_FUNCTION (clEnqueueWriteImage);
  CL_LOAD_FUNCTION (clEnqueueCopyImage);
  CL_LOAD_FUNCTION (clEnqueueCopyImageToBuffer);
  CL_LOAD_FUNCTION (clEnqueueCopyBufferToImage);
  CL_LOAD_FUNCTION (clEnqueueFillImage);
  CL_LOAD_FUNCTION (clEnqueueMapImage);
  CL_LOAD_FUNCTION (clGetImageInfo);
  CL_LOAD_FUNCTION (clGetSupportedImageFormats);

  /* Memory Objects */
  CL_LOAD_FUNCTION (clEnqueueUnmapMemObject);
  CL_LOAD_FUNCTION (clEnqueueMigrateMemObjects);
  CL_LOAD_FUNCTION (clGetMemObjectInfo);
  CL_LOAD_FUNCTION (clRetainMemObject);
  CL_LOAD_FUNCTION (clReleaseMemObject);
  CL_LOAD_FUNCTION (clSetMemObjectDestructorCallback);

  /* Sampler Objects */
#ifndef __APPLE__
  CL_LOAD_FUNCTION (clCreateSamplerWithProperties);
#endif
  CL_LOAD_FUNCTION (clReleaseSampler);
  CL_LOAD_FUNCTION (clRetainSampler);
  CL_LOAD_FUNCTION (clGetSamplerInfo);

  /* Program Objects */
  CL_LOAD_FUNCTION (clBuildProgram);
  CL_LOAD_FUNCTION (clCompileProgram);
  CL_LOAD_FUNCTION (clCreateProgramWithSource);
  CL_LOAD_FUNCTION (clCreateProgramWithBinary);
  CL_LOAD_FUNCTION (clCreateProgramWithBuiltInKernels);
#ifndef __APPLE__
  CL_LOAD_FUNCTION (clCreateProgramWithIL);
#endif
  CL_LOAD_FUNCTION (clGetProgramBuildInfo);
  CL_LOAD_FUNCTION (clGetProgramInfo);
  CL_LOAD_FUNCTION (clLinkProgram);
  CL_LOAD_FUNCTION (clReleaseProgram);
  CL_LOAD_FUNCTION (clRetainProgram);
#ifndef __APPLE__
  CL_LOAD_FUNCTION (clSetProgramSpecializationConstant);
#endif
  CL_LOAD_FUNCTION (clUnloadPlatformCompiler);

  /* Kernel Objects */
#ifndef __APPLE__
  CL_LOAD_FUNCTION (clCloneKernel);
#endif
  CL_LOAD_FUNCTION (clCreateKernel);
  CL_LOAD_FUNCTION (clCreateKernelsInProgram);
  CL_LOAD_FUNCTION (clGetKernelInfo);
  CL_LOAD_FUNCTION (clGetKernelArgInfo);
#ifndef __APPLE__
  CL_LOAD_FUNCTION (clGetKernelSubGroupInfo);
#endif
  CL_LOAD_FUNCTION (clGetKernelWorkGroupInfo);
  CL_LOAD_FUNCTION (clReleaseKernel);
  CL_LOAD_FUNCTION (clRetainKernel);
  CL_LOAD_FUNCTION (clSetKernelArg);
#ifndef __APPLE__
  CL_LOAD_FUNCTION (clSetKernelArgSVMPointer);
  CL_LOAD_FUNCTION (clSetKernelExecInfo);
#endif

  /* Executing Kernels */
  CL_LOAD_FUNCTION (clEnqueueNDRangeKernel);
  CL_LOAD_FUNCTION (clEnqueueNativeKernel);

  /* Event Objects */
  CL_LOAD_FUNCTION (clCreateUserEvent);
  CL_LOAD_FUNCTION (clGetEventInfo);
  CL_LOAD_FUNCTION (clReleaseEvent);
  CL_LOAD_FUNCTION (clRetainEvent);
  CL_LOAD_FUNCTION (clSetEventCallback);
  CL_LOAD_FUNCTION (clSetUserEventStatus);
  CL_LOAD_FUNCTION (clWaitForEvents);

  /* Markers, Barriers, and Waiting */
  CL_LOAD_FUNCTION (clEnqueueBarrierWithWaitList);
  CL_LOAD_FUNCTION (clEnqueueMarkerWithWaitList);

  /* Profiling Operations on Memory Objects and Kernels */
#ifndef __APPLE__
  CL_LOAD_FUNCTION (clGetDeviceAndHostTimer);
#endif
  CL_LOAD_FUNCTION (clGetEventProfilingInfo);
#ifndef __APPLE__
  CL_LOAD_FUNCTION (clGetHostTimer);
#endif

  /* Flush and Finish */
  CL_LOAD_FUNCTION (clFlush);
  CL_LOAD_FUNCTION (clFinish);

  /* Pipes */
#ifndef __APPLE__
  CL_LOAD_FUNCTION (clCreatePipe);
  CL_LOAD_FUNCTION (clGetPipeInfo);
#endif

  /* Shared Virtual Memory (SVM) */
#ifndef __APPLE__
  CL_LOAD_FUNCTION (clSVMAlloc);
  CL_LOAD_FUNCTION (clSVMFree);
  CL_LOAD_FUNCTION (clEnqueueSVMFree);
  CL_LOAD_FUNCTION (clEnqueueSVMMap);
  CL_LOAD_FUNCTION (clEnqueueSVMMemcpy);
  CL_LOAD_FUNCTION (clEnqueueSVMMemFill);
  CL_LOAD_FUNCTION (clEnqueueSVMMigrateMem);
  CL_LOAD_FUNCTION (clEnqueueSVMUnmap);
#endif

  /* Optional Extensions */
  CL_LOAD_FUNCTION (clGetExtensionFunctionAddressForPlatform);

  return TRUE;
}

#undef CL_LOAD_FUNCTION
#undef CL_LOAD_EXTENSION_FUNCTION

static gboolean
gegl_cl_init_load_device_info (cl_platform_id platform,
                               cl_device_id   device,
                               cl_device_type requested_device_type)
{
  cl_int err = CL_SUCCESS;

  if (device)
    {
      /* Get platform from device */
      err = gegl_clGetDeviceInfo (device, CL_DEVICE_PLATFORM, sizeof (cl_platform_id), &platform, NULL);
      if (err != CL_SUCCESS)
        {
          GEGL_NOTE (GEGL_DEBUG_OPENCL, "Could not create platform");
          return FALSE;
        }
    }
  else
    {
      cl_platform_id *platforms = NULL;
      cl_uint num_platforms = 0;

      if (!requested_device_type)
        requested_device_type = CL_DEVICE_TYPE_DEFAULT;

      err = gegl_clGetPlatformIDs (0, NULL, &num_platforms);
      if (err != CL_SUCCESS)
        {
          GEGL_NOTE (GEGL_DEBUG_OPENCL, "Could not create platform");
          return FALSE;
        }

      if (platform)
        {
          platforms = g_new (cl_platform_id, 1);
          num_platforms = 1;
          platforms[0] = platform;
        }
      else
        {
          platforms = g_new (cl_platform_id, num_platforms);
          err = gegl_clGetPlatformIDs (num_platforms, platforms, NULL);
        }

      if (err != CL_SUCCESS)
        {
          GEGL_NOTE (GEGL_DEBUG_OPENCL, "Could not create platform");
          g_free (platforms);
          return FALSE;
        }

      for (int platform_idx = 0; platform_idx < num_platforms; platform_idx++)
        {
          platform = platforms[platform_idx];
          err = gegl_clGetDeviceIDs (platform, requested_device_type, 1, &device, NULL);
          if (err == CL_SUCCESS) {
            cl_bool tmp_image_support = FALSE;
            err = gegl_clGetDeviceInfo (device, CL_DEVICE_IMAGE_SUPPORT, sizeof(cl_bool), &tmp_image_support, NULL);
            if (err == CL_SUCCESS && tmp_image_support == FALSE)
              continue;
            break;
          }
        }

      g_free (platforms);

      if (err != CL_SUCCESS)
        {
          GEGL_NOTE (GEGL_DEBUG_OPENCL, "Could not create device: %s", gegl_cl_errstring (err));
          return FALSE;
        }
    }

  cl_state.platform = platform;
  cl_state.device = device;

  gegl_clGetPlatformInfo (platform, CL_PLATFORM_NAME,       sizeof(cl_state.platform_name),    cl_state.platform_name,    NULL);
  gegl_clGetPlatformInfo (platform, CL_PLATFORM_VERSION,    sizeof(cl_state.platform_version), cl_state.platform_version, NULL);
  gegl_clGetPlatformInfo (platform, CL_PLATFORM_EXTENSIONS, sizeof(cl_state.platform_ext),     cl_state.platform_ext,     NULL);

  gegl_clGetDeviceInfo (device, CL_DEVICE_NAME, sizeof(cl_state.device_name), cl_state.device_name, NULL);
  gegl_clGetDeviceInfo (device, CL_DEVICE_VERSION, sizeof(cl_state.device_version), cl_state.device_version, NULL);

  gegl_clGetDeviceInfo (device, CL_DEVICE_IMAGE_SUPPORT,      sizeof(cl_bool),  &cl_state.image_support,    NULL);
  gegl_clGetDeviceInfo (device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &cl_state.max_mem_alloc,    NULL);
  gegl_clGetDeviceInfo (device, CL_DEVICE_LOCAL_MEM_SIZE,     sizeof(cl_ulong), &cl_state.local_mem_size,   NULL);
  gegl_clGetDeviceInfo (device, CL_DEVICE_SINGLE_FP_CONFIG,   sizeof(cl_device_fp_config), &cl_state.fp_config, NULL);

  cl_state.platform_v3 = (strstr(cl_state.platform_version, "OpenCL 3.0") != NULL);
  cl_state.compiler_v3 = FALSE;
  if (cl_state.platform_v3)
    {
      size_t          ret_size = 0;
      guint           num_items;
      cl_name_version cl_c_versions[10];

      if (gegl_clGetDeviceInfo (device, CL_DEVICE_OPENCL_C_ALL_VERSIONS,
                                sizeof(cl_c_versions), cl_c_versions, &ret_size) != CL_SUCCESS)
        ret_size = 0;
      num_items = MIN (ret_size / sizeof(cl_name_version), G_N_ELEMENTS (cl_c_versions));

      for (int i = 0; i < num_items; i++)
        {
          if (CL_VERSION_MAJOR (cl_c_versions[i].version) == 3 &&
              CL_VERSION_MINOR (cl_c_versions[i].version) == 0)
            {
              cl_state.compiler_v3 = TRUE;
              break;
            }
        }
    }

  cl_state.iter_width  = 4096;
  cl_state.iter_height = 4096;

  while (cl_state.iter_width * cl_state.iter_height * 16 > cl_state.max_mem_alloc)
    {
      if (cl_state.iter_height < cl_state.iter_width)
        cl_state.iter_width  /= 2;
      else
        cl_state.iter_height /= 2;
    }

  cl_state.iter_width  /= 2;

  GEGL_NOTE (GEGL_DEBUG_OPENCL, "Platform Name: %s",       cl_state.platform_name);
  GEGL_NOTE (GEGL_DEBUG_OPENCL, "Version: %s",             cl_state.platform_version);
  GEGL_NOTE (GEGL_DEBUG_OPENCL, "Extensions: %s",          cl_state.platform_ext);
  GEGL_NOTE (GEGL_DEBUG_OPENCL, "Default Device Name: %s", cl_state.device_name);
  GEGL_NOTE (GEGL_DEBUG_OPENCL, "Default Device Version: %s", cl_state.device_version);
  if (cl_state.platform_v3 && cl_state.compiler_v3)
    GEGL_NOTE (GEGL_DEBUG_OPENCL, "Default Device OpenCL C supports 3.0");
  else if (cl_state.platform_v3)
    GEGL_NOTE (GEGL_DEBUG_OPENCL, "Default Device supports 3.0 but OpenCL C does not");
  if ((cl_state.fp_config & CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT) == 0)
    GEGL_NOTE (GEGL_DEBUG_OPENCL, "Device doesn't support -cl-fp32-correctly-rounded-divide-sqrt");
  GEGL_NOTE (GEGL_DEBUG_OPENCL, "Max Alloc: %lu bytes",    (unsigned long)cl_state.max_mem_alloc);
  GEGL_NOTE (GEGL_DEBUG_OPENCL, "Local Mem: %lu bytes",    (unsigned long)cl_state.local_mem_size);
  GEGL_NOTE (GEGL_DEBUG_OPENCL, "Iteration size: (%lu, %lu)",
                                (long unsigned int)cl_state.iter_width,
                                (long unsigned int)cl_state.iter_height);

  return TRUE;
}

static gboolean
gegl_cl_init_common (cl_device_type requested_device_type)
{
  cl_int err;

  if (cl_state.hard_disable)
    {
      GEGL_NOTE (GEGL_DEBUG_OPENCL, "OpenCL is disabled");
      return FALSE;
    }

  if (!cl_state.is_loaded)
    {
#ifndef __APPLE__
      cl_queue_properties queue_props[] = { CL_QUEUE_PROPERTIES, 0, 0 };
#else
      cl_command_queue_properties command_queue_flags = 0;
#endif
      cl_context          ctx           = NULL;

      if (!gegl_cl_init_load_functions ())
        return FALSE;

      if (!gegl_cl_init_load_device_info (NULL, NULL, requested_device_type))
        return FALSE;
      ctx = gegl_clCreateContext (NULL, 1, &cl_state.device, NULL, NULL, &err);

      if (cl_state.image_support)
        {
          GEGL_NOTE (GEGL_DEBUG_OPENCL, "Image Support OK");
        }
      else
        {
          if (ctx)
            gegl_clReleaseContext (ctx);

          GEGL_NOTE (GEGL_DEBUG_OPENCL, "Image Support Error");
          return FALSE;
        }

      cl_state.ctx = ctx;

#ifdef __APPLE__
      command_queue_flags = 0;
#endif
      if (cl_state.enable_profiling)
#ifndef __APPLE__
        queue_props[1] = CL_QUEUE_PROFILING_ENABLE;
#else
        command_queue_flags |= CL_QUEUE_PROFILING_ENABLE;
#endif

#ifndef __APPLE__
      cl_state.cq = gegl_clCreateCommandQueueWithProperties (cl_state.ctx, cl_state.device, (const cl_queue_properties *)&queue_props, &err);
#else
      cl_state.cq = gegl_clCreateCommandQueue (cl_state.ctx, cl_state.device, command_queue_flags, &err);
#endif

      if (err != CL_SUCCESS)
        {
          GEGL_NOTE (GEGL_DEBUG_OPENCL, "Could not create command queue: %s", gegl_cl_errstring (err));
          return FALSE;
        }

      _gegl_cl_is_accelerated = TRUE;
      cl_state.is_loaded = TRUE;

      /* XXX: this dict is being leaked */
      cl_program_hash[FALSE] = g_hash_table_new (g_str_hash, g_str_equal);
      cl_program_hash[TRUE] = g_hash_table_new (g_str_hash, g_str_equal);

      gegl_cl_color_compile_kernels ();

      GEGL_NOTE (GEGL_DEBUG_OPENCL, "OK");
    }

  if (cl_state.is_loaded)
      _gegl_cl_is_accelerated = TRUE;

  {
    gegl_buffer_ext_flush = (void*)gegl_buffer_cl_cache_flush;
    gegl_buffer_ext_invalidate = (void*)gegl_buffer_cl_cache_invalidate;
    gegl_tile_handler_cache_ext_flush = (void*)gegl_buffer_cl_cache_flush2;
  }

  return TRUE;
}

static GeglClRunData *
compile_and_build (const char *program_source,
                   const char *kernel_name[],
                   gboolean    optimize)
{
  GeglClRunData *cached;
  GeglClRunData *cl_data;
  const char *sources[] = {random_cl_source, program_source};
  size_t lengths[2];
  gchar   options[96];
  gchar  *msg = NULL;
  size_t  log_size = 0;
  cl_int  cl_err;
  cl_int  build_errcode;
  guint   kernel_n = 0;

  if (!gegl_cl_is_accelerated ())
    return NULL;

  g_mutex_lock (&cl_program_mutex);
  cached = g_hash_table_lookup (cl_program_hash[optimize], program_source);
  g_mutex_unlock (&cl_program_mutex);
  if (cached)
    return cached;

  while (kernel_name[kernel_n] != NULL)
    kernel_n++;

  lengths[0] = strlen (random_cl_source);
  lengths[1] = strlen (program_source);
  cl_data = g_new0 (GeglClRunData, 1);
  cl_data->program = gegl_clCreateProgramWithSource (gegl_cl_get_context (), 2, sources,
                                                     lengths, &cl_err);
  CL_CHECK;

  g_snprintf (options, sizeof (options), "-cl-std=%s%s%s",
              cl_state.compiler_v3 ? "CL3.0" : "CL1.2",
              optimize ? "" : " -cl-opt-disable",
              cl_state.fp_config & CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT ?
                " -cl-fp32-correctly-rounded-divide-sqrt" : "");
  build_errcode = gegl_clBuildProgram (cl_data->program, 0, NULL,
                                       options, NULL, NULL);
  cl_err = gegl_clGetProgramBuildInfo (cl_data->program, gegl_cl_get_device (),
                                        CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
  CL_CHECK;

  msg = g_new0 (gchar, log_size + 1);
  if (log_size)
    {
      cl_err = gegl_clGetProgramBuildInfo (cl_data->program, gegl_cl_get_device (),
                                            CL_PROGRAM_BUILD_LOG, log_size, msg, NULL);
      CL_CHECK;
    }

  if (build_errcode != CL_SUCCESS)
    {
      GEGL_NOTE (GEGL_DEBUG_OPENCL, "Build Error: %s\n%s",
                                    gegl_cl_errstring (build_errcode), msg);
      g_warning ("%s\n%s\n", gegl_cl_errstring (build_errcode), msg);
      goto error;
    }

  g_strchug (msg);
  if (*msg)
    GEGL_NOTE (GEGL_DEBUG_OPENCL, "Compiling successful\n%s", msg);
  else
    GEGL_NOTE (GEGL_DEBUG_OPENCL, "Compiling successful");
  g_clear_pointer (&msg, g_free);

  cl_data->kernel = g_new0 (cl_kernel, kernel_n);
  cl_data->work_group_size = g_new (size_t, kernel_n);

  for (guint i = 0; i < kernel_n; i++)
    {
      cl_data->kernel[i] = gegl_clCreateKernel (cl_data->program,
                                                kernel_name[i], &cl_err);
      CL_CHECK;

      cl_err = gegl_clGetKernelWorkGroupInfo (cl_data->kernel[i],
                                              gegl_cl_get_device (),
                                              CL_KERNEL_WORK_GROUP_SIZE,
                                              sizeof (size_t),
                                              &cl_data->work_group_size[i],
                                              NULL);
      CL_CHECK;
    }

  /* Don't hold the cache lock while compiling or reporting a driver error.
   * A render can compile a custom kernel while graph preparation compiles
   * a class kernel, and only the first successful build should be cached. */
  g_mutex_lock (&cl_program_mutex);
  cached = g_hash_table_lookup (cl_program_hash[optimize], program_source);
  if (!cached)
    {
      g_hash_table_insert (cl_program_hash[optimize], g_strdup (program_source), cl_data);
      g_mutex_unlock (&cl_program_mutex);
      return cl_data;
    }
  g_mutex_unlock (&cl_program_mutex);

error:
  g_free (msg);
  if (cl_data->kernel)
    for (guint i = 0; i < kernel_n; i++)
      if (cl_data->kernel[i])
        gegl_clReleaseKernel (cl_data->kernel[i]);
  g_free (cl_data->kernel);
  g_free (cl_data->work_group_size);
  if (cl_data->program)
    gegl_clReleaseProgram (cl_data->program);
  g_free (cl_data);
  return cached;
}

GeglClRunData *
gegl_cl_compile_and_build (const char *program_source,
                           const char *kernel_name[])
{
  /* SNN mean can change output with compiler optimizations enabled. */
  return compile_and_build (program_source, kernel_name, FALSE);
}

GeglClRunData *
gegl_cl_compile_and_build_optimized (const char *program_source,
                                     const char *kernel_name[])
{
  return compile_and_build (program_source, kernel_name, TRUE);
}

/* A small pool of device buffers, so that the buffer iterators and the
 * operations running one after another do not create and release (and on
 * many drivers, map and fault in) new device memory for every chunk.
 *
 * Buffers are matched on their flags and a size between the requested size
 * and twice that; a buffer's content is undefined when it is handed out,
 * as with a new one.  The pool keeps at most GEGL_CL_MEM_POOL_MAX_BUFFERS
 * buffers and GEGL_CL_MEM_POOL_MAX_BYTES bytes, releasing the least
 * recently returned buffers first.
 */
#define GEGL_CL_MEM_POOL_MAX_BUFFERS 8
#define GEGL_CL_MEM_POOL_MAX_BYTES   ((size_t) 160 * 1024 * 1024)

typedef struct
{
  cl_mem       mem;
  cl_mem_flags flags;
  size_t       size;
} GeglClPooledMem;

static GMutex  cl_mem_pool_mutex;
static GQueue  cl_mem_pool        = G_QUEUE_INIT; /* oldest first */
static size_t  cl_mem_pool_bytes  = 0;

cl_mem
gegl_cl_mem_pool_get (cl_mem_flags  flags,
                      size_t        size,
                      cl_int       *errcode_ret)
{
  GeglClPooledMem *best = NULL;

  g_mutex_lock (&cl_mem_pool_mutex);

  for (GList *iter = cl_mem_pool.head; iter; iter = iter->next)
    {
      GeglClPooledMem *pooled = iter->data;

      if (pooled->flags == flags &&
          pooled->size  >= size  &&
          pooled->size / 2 <= size &&
          (! best || pooled->size < best->size))
        best = pooled;
    }

  if (best)
    {
      g_queue_remove (&cl_mem_pool, best);
      cl_mem_pool_bytes -= best->size;
    }

  g_mutex_unlock (&cl_mem_pool_mutex);

  if (best)
    {
      cl_mem mem = best->mem;

      g_slice_free (GeglClPooledMem, best);

      if (errcode_ret)
        *errcode_ret = CL_SUCCESS;

      return mem;
    }

  return gegl_clCreateBuffer (gegl_cl_get_context (), flags, size, NULL,
                              errcode_ret);
}

void
gegl_cl_mem_pool_put (cl_mem mem)
{
  GeglClPooledMem *pooled;
  GList           *evict = NULL;
  cl_mem_flags     flags = 0;
  size_t           size  = 0;
  void            *host_ptr = NULL;

  if (! mem)
    return;

  /* only pool plain device buffers we can recreate from flags and size */
  if (gegl_clGetMemObjectInfo (mem, CL_MEM_FLAGS, sizeof (flags), &flags, NULL) != CL_SUCCESS ||
      gegl_clGetMemObjectInfo (mem, CL_MEM_SIZE, sizeof (size), &size, NULL) != CL_SUCCESS ||
      gegl_clGetMemObjectInfo (mem, CL_MEM_HOST_PTR, sizeof (host_ptr), &host_ptr, NULL) != CL_SUCCESS ||
      host_ptr != NULL ||
      size > GEGL_CL_MEM_POOL_MAX_BYTES)
    {
      gegl_clReleaseMemObject (mem);
      return;
    }

  pooled = g_slice_new (GeglClPooledMem);
  pooled->mem   = mem;
  pooled->flags = flags;
  pooled->size  = size;

  g_mutex_lock (&cl_mem_pool_mutex);

  g_queue_push_tail (&cl_mem_pool, pooled);
  cl_mem_pool_bytes += size;

  while (cl_mem_pool_bytes > GEGL_CL_MEM_POOL_MAX_BYTES ||
         cl_mem_pool.length > GEGL_CL_MEM_POOL_MAX_BUFFERS)
    {
      GeglClPooledMem *oldest = g_queue_pop_head (&cl_mem_pool);

      cl_mem_pool_bytes -= oldest->size;
      evict = g_list_prepend (evict, oldest);
    }

  g_mutex_unlock (&cl_mem_pool_mutex);

  for (GList *iter = evict; iter; iter = iter->next)
    {
      GeglClPooledMem *oldest = iter->data;

      gegl_clReleaseMemObject (oldest->mem);
      g_slice_free (GeglClPooledMem, oldest);
    }
  g_list_free (evict);
}

void
gegl_cl_mem_pool_clear (void)
{
  GeglClPooledMem *pooled;

  g_mutex_lock (&cl_mem_pool_mutex);

  while ((pooled = g_queue_pop_head (&cl_mem_pool)))
    {
      gegl_clReleaseMemObject (pooled->mem);
      g_slice_free (GeglClPooledMem, pooled);
    }
  cl_mem_pool_bytes = 0;

  g_mutex_unlock (&cl_mem_pool_mutex);
}

void
gegl_cl_cleanup (void)
{
  cl_int err;

  gegl_cl_mem_pool_clear ();

  err = gegl_cl_random_cleanup ();
  if (err != CL_SUCCESS)
    GEGL_NOTE (GEGL_DEBUG_OPENCL, "Could not free cl_random_data: %s", gegl_cl_errstring (err));
}
