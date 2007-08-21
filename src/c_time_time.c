#include "c_time_time.h"
#include <sys/time.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

double time_time(void) {
  struct timeval t;

  if (gettimeofday(&t, (struct timezone *)NULL) == 0)
    return (double)t.tv_sec + t.tv_usec*0.000001;
  fprintf(stderr,"Error getting time file %s (line %d), exiting.\n",__FILE__,__LINE__);
  exit(1);
}
