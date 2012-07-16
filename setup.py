from distutils.core import setup
from distutils.extension import Extension

lbfgs_sources = ['liblbfgs/lbfgs.c']
creg_sources = ('fdict.cc', 'dict.cc', 'gzstream.cc', 'filelib.cc',
                'json_feature_map_lexer.cc')
creg_sources = ['creg/'+fn for fn in creg_sources]

ext_modules = [
    Extension(name='_creg',
        sources=['python/_creg.cpp'] + lbfgs_sources + creg_sources,
        language='c++', 
        include_dirs=['.', 'creg', 'python'],
        libraries=['boost_program_options-mt', 'z', 'stdc++'])
]

setup(
    name='creg',
    ext_modules=ext_modules,
    packages=['creg']
)
