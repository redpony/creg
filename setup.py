from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

lbfgs_sources = ['liblbfgs/lbfgs.c']
creg_sources = ('fdict.cc', 'dict.cc', 'gzstream.cc', 'filelib.cc',
                'json_feature_map_lexer.cc')
creg_sources = ['creg/'+fn for fn in creg_sources]

ext_modules = [
    Extension(name='_creg',
        sources=['python/_creg.pyx'] + lbfgs_sources + creg_sources,
        language='c++', 
        include_dirs=['.', 'creg', 'python'],
        libraries=['boost_program_options-mt', 'z'])
]

setup(
    name='creg',
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules,
    package_dir={'creg': 'python'},
    packages=['creg']
)
