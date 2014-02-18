bin_PROGRAMS = \
  creg

creg_SOURCES = \
  creg.cc \
  dict.cc \
  dict.h \
  fdict.cc \
  fdict.h \
  feature_map.h \
  filelib.cc \
  filelib.h \
  gzstream.cc \
  gzstream.h \
  json_feature_map_lexer.cc \
  json_feature_map_lexer.h \
  logval.h \
  prob.h \
  wordid.h

creg_LDADD = ../liblbfgs/liblbfgs.a

AM_CPPFLAGS = -DBOOST_TEST_DYN_LINK -W -Wall -Wno-sign-compare -I$(top_srcdir)

