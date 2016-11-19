#include <stdio.h>

/* When we don't have Python GIL, we can't raise exception, so let's just quit */
#define SET_ERR( errval )						\
  {  printf("SET_ERR(%d) called, %s: %d\n",errval,__FILE__,__LINE__);	\
    exit(errval);							\
  }
