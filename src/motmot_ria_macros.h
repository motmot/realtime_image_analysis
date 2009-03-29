#include <stdio.h>

#include <fwBase.h>
#include <fwImage.h>

#define IMPOS8u(im,step,bot,left) ((im)+((bot)*(step))+(left))
#define IMPOS32f(im,step,bot,left) ((im)+((bot)*(step/4))+(left))
#define CHK_NOGIL( errval ) \
  if ( (errval) )	    \
    { \
      fprintf(stderr,"IPP ERROR %d: in file %s (line %d), exiting because I may not have GIL\n",(errval),__FILE__,__LINE__); \
      exit(1); \
    }

#define CHK_FIC_NOGIL( errval ) \
  if ( (errval) )	    \
    { \
      fprintf(stderr,"FIC ERROR %d: in file %s (line %d), exiting because I may not have GIL\n",(errval),__FILE__,__LINE__); \
      exit(1); \
    }

/* When we don't have Python GIL, we can't raise exception, so let's just quit */
#define SET_ERR( errval )						\
  {  printf("SET_ERR(%d) called, %s: %d\n",errval,__FILE__,__LINE__);	\
    exit(errval);							\
  }


#if defined(__GNUC__)
#if defined(_WIN32)
int __security_cookie;
void __fastcall __security_check_cookie(void *stackAddress){}

long long _allmul( long long a, long long b) {return a*b;}
#endif
#endif
