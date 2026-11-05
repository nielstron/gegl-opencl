/* SPDX-License-Identifier: LGPL-3.0-or-later
 * Deterministic regression tests for c2g's OpenCL math, independent of its
 * stochastic sampling. Run with the path to opencl/c2g.cl as argv[1].
 */
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#endif
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(call) do { cl_int e = (call); if (e != CL_SUCCESS) { \
  fprintf(stderr, "%s: OpenCL error %d\n", #call, e); exit(1); } } while (0)
#define ANGLES 95273
#define RADII 29537

int main(int argc, char **argv)
{
  cl_platform_id platform;
  cl_device_id device;
  cl_int err;
  cl_uint count, nargs;
  if (argc != 2) return 1;
  if (clGetPlatformIDs(1, &platform, &count) != CL_SUCCESS || count == 0)
    return 77;
  if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL) != CL_SUCCESS)
    return 77;
  cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  CHECK(err);
  cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
  CHECK(err);
  FILE *file = fopen(argv[1], "rb");
  if (!file) return 1;
  fseek(file, 0, SEEK_END);
  size_t length = ftell(file);
  rewind(file);
  char *source = malloc(length + 1);
  if (fread(source, 1, length, file) != length) return 1;
  source[length] = 0;
  fclose(file);
  cl_program program = clCreateProgramWithSource(context, 1,
      (const char **)&source, &length, &err);
  CHECK(err);
  err = clBuildProgram(program, 1, &device, "", NULL, NULL);
  if (err != CL_SUCCESS) {
    char log[16384];
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
    fprintf(stderr, "%s\n", log);
    return 1;
  }
  cl_kernel kernel = clCreateKernel(program, "c2g", &err);
  CHECK(err);
  CHECK(clGetKernelInfo(kernel, CL_KERNEL_NUM_ARGS, sizeof(nargs), &nargs, NULL));
  float *cosines = malloc(ANGLES * sizeof(float));
  float *sines = malloc(ANGLES * sizeof(float));
  float *radii = malloc(RADII * sizeof(float));
  const float cs[] = {1, 0, -1, 0}, sn[] = {0, 1, 0, -1};
  for (int i = 0; i < ANGLES; ++i) { cosines[i] = cs[i % 4]; sines[i] = sn[i % 4]; }
  for (int i = 0; i < RADII; ++i) radii[i] = 1;
  float pixels[25 * 4];
  cl_mem input = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(pixels), NULL, &err);
  CHECK(err);
  cl_mem output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 2 * sizeof(float), NULL, &err);
  CHECK(err);
  cl_mem r = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      RADII * sizeof(float), radii, &err); CHECK(err);
  cl_mem c = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      ANGLES * sizeof(float), cosines, &err); CHECK(err);
  cl_mem s = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      ANGLES * sizeof(float), sines, &err); CHECK(err);
  int width = 5, height = 5, radius = 2, samples = 4, iterations = 1;
#define ARG(i, v) CHECK(clSetKernelArg(kernel, i, sizeof(v), &(v)))
  ARG(0, input); ARG(1, width); ARG(2, height); ARG(3, r); ARG(4, c);
  ARG(5, s); ARG(6, output); ARG(7, radius); ARG(8, samples); ARG(9, iterations);
  int failures = 0;
  for (int uniform = 0; uniform <= 1; ++uniform) {
    for (int shadows = 0; shadows <= 1; ++shadows) {
      for (int i = 0; i < 25; ++i) {
        float value = uniform ? 0.25f : (i % 5) / 4.0f;
        for (int channel = 0; channel < 3; ++channel) pixels[i * 4 + channel] = value;
        pixels[i * 4 + 3] = 0.37f;
      }
      CHECK(clEnqueueWriteBuffer(queue, input, CL_TRUE, 0, sizeof(pixels), pixels, 0, NULL, NULL));
      /* Also run on the old kernel, to demonstrate both regressions before patching. */
      if (nargs > 10) ARG(10, shadows);
      const size_t size[] = {1, 1};
      CHECK(clEnqueueNDRangeKernel(queue, kernel, 2, NULL, size, NULL, 0, NULL, NULL));
      float result[2];
      CHECK(clEnqueueReadBuffer(queue, output, CL_TRUE, 0, sizeof(result), result, 0, NULL, NULL));
      float expected = uniform && !shadows ? 1.0f : 0.5f;
      int ok = fabsf(result[0] - expected) < 1e-5f && fabsf(result[1] - 0.37f) < 1e-5f;
      printf("uniform=%d enhance-shadows=%d: gray=%g expected=%g alpha=%g %s\n",
          uniform, shadows, result[0], expected, result[1], ok ? "PASS" : "FAIL");
      failures += !ok;
    }
  }
  clReleaseMemObject(input); clReleaseMemObject(output);
  clReleaseMemObject(r); clReleaseMemObject(c); clReleaseMemObject(s);
  clReleaseKernel(kernel); clReleaseProgram(program);
  clReleaseCommandQueue(queue); clReleaseContext(context);
  free(source); free(cosines); free(sines); free(radii);
  return failures ? 1 : 0;
}
