
opencl_tests = [
  'bilateral-filter',
  'box-blur',
  'brightness-contrast',
  'color-enhance',
  'color-temperature',
  'deinterlace',
  'edge-sobel',
  'edge-sobel-both-axes',
  'emboss',
  'focus-blur',
  'gaussian-blur',
  'gaussian-blur-iir',
  'invert-linear',
  'lens-blur',
  'lens-distortion',
  'levels',
  'mean-curvature-blur',
  'median-blur',
  'mono-mixer',
  'motion-blur',
  'motion-blur-circular',
  'motion-blur-zoom',
  'newsprint',
  'noise-cie-lch',
  'noise-pick',
  'noise-reduction',
  'noise-rgb',
  'oilify',
  'opacity',
  'pixelize',
  'red-eye-removal',
  'snn-mean',
  'svg-src-over',
  'texturize-canvas',
  'threshold',
  'value-invert',
  'vignette',
]

foreach _test : opencl_tests
  test(_test.underscorify(),
    test_runner,
    args: [
      '--test-name', _test,
      '--output-dir', meson.current_build_dir() / 'output',
      '--reference-path', meson.current_build_dir() / 'reference',
      '--endian', host_machine.endian(),
      '--generate-reference',
      '--input-file', files('@0@.xml'.format(_test)),
      '--gegl-exe', gegl_bin,
      '--imgcmp-exe', gegl_imgcmp,
      '--with-opencl',
    ],
    env: gegl_test_env,
    suite: 'opencl',
    timeout: 60,
    is_parallel: gegl_test_parallel,
    )
endforeach

# Exercise the actual kernel with deterministic sample tables. This isolates
# projection math from c2g's stochastic CPU/GPU sampling sequences.
if host_machine.system() == 'darwin'
  c2g_opencl = dependency('appleframeworks', modules: ['OpenCL'], required: false)
else
  c2g_opencl = dependency('OpenCL', required: false)
endif
if c2g_opencl.found()
  test('c2g-kernel',
    executable('test-c2g-kernel', 'test-c2g-kernel.c',
      dependencies: [c2g_opencl, meson.get_compiler('c').find_library('m', required: false)]),
    args: [meson.project_source_root() / 'operations/opencl/c2g.cl'],
    suite: 'opencl',
    timeout: 60,
  )
endif
