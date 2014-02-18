noinst_LIBRARIES = liblbfgs.a

liblbfgs_a_SOURCES = \
  arithmetic_ansi.h \
  arithmetic_sse_double.h \
  arithmetic_sse_float.h \
  lbfgs++.h \
  lbfgs.c \
  lbfgs.h

################################################################
# do NOT NOT NOT add any other -I includes NO NO NO NO NO ######
AM_CPPFLAGS = -DBOOST_TEST_DYN_LINK -W -Wall -I. -I$(top_srcdir)
################################################################

