from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

lbfgs_sources = ['liblbfgs/lbfgs.c']
creg_sources = ('fdict.cc', 'dict.cc', 'gzstream.cc', 'filelib.cc', 'json_feature_map_lexer.cc')
creg_sources = ['creg/'+fn for fn in creg_sources]

ext_modules = [
    Extension(name='creg',
        sources=['python/creg.pyx'] + lbfgs_sources + creg_sources,
        language='C++', 
        include_dirs=['.', 'creg'],
        libraries=['boost_program_options-mt'])
]

setup(
    name='creg',
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules
)
