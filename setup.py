from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [
    Extension(name='creg',
        sources=['python/creg.pyx', 'creg/json_feature_map_lexer.cc'],
        language='C++', 
        include_dirs=['.'],
        extra_objects = ['dist/lib/libcregutils.a', 'dist/lib/liblbfgs.a'],
        libraries=['boost_program_options-mt'])
]

setup(
    name='creg',
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules
)
