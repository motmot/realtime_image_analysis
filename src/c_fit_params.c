/* $Id: c_fit_params.c 1614 2007-02-28 01:17:56Z astraw $ */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "c_fit_params.h"
#include <fwBase.h>

#define CHK( a ) (a == fwStsNoErr? 1:0)
#define CHKCV( a, errtype ) {                   \
    (a);                                        \
    if (0) {                                    \
      return errtype;                           \
    }                                           \
  }

CvMat fi2cv_view(unsigned char *img, int img_step, int width, int height) {
  // See cvMat definition in cxtypes.h

  CvMat result;
  int type;

  type = CV_8U;
  type = CV_MAT_TYPE(type);
  result.type = CV_MAT_MAGIC_VAL | CV_MAT_CONT_FLAG | type;
  result.cols = width;
  result.rows = height;
  result.step = img_step;
  result.data.ptr = img;
  return result;
}

/****************************************************************
** fit_params ***************************************************
****************************************************************/
CFitParamsReturnType fit_params( CvMoments *pState,
                                 double *x0, double *y0,
                                 double *Mu00,
                                 double *Uu11, double *Uu20, double *Uu02,
                                 int width, int height,
                                 unsigned char *img, int img_step )
{
  double Mu10, Mu01;
  double val;
  CvMat arr;

  arr = fi2cv_view(img, img_step, width, height);

  /* get moments */
  int binary=0; // not a binary image?
  CHKCV( cvMoments( &arr, pState, binary ), CFitParamsOtherError);

  /* calculate center of gravity from spatial moments */
  CHKCV( val = cvGetSpatialMoment( pState, 0, 0), CFitParamsOtherError);
  *Mu00 = val;

  CHKCV( Mu10 = cvGetSpatialMoment( pState, 1, 0), CFitParamsOtherError);
  CHKCV( Mu01 = cvGetSpatialMoment( pState, 0, 1), CFitParamsOtherError);

  if( *Mu00 != 0.0 )
  {
    *x0 = Mu10 / *Mu00;
    *y0 = Mu01 / *Mu00;
    /* relative to ROI origin */
  }
  else
  {
    return CFitParamsZeroMomentError;
  }

  /* calculate blob orientation from central moments */
  CHKCV( val = cvGetCentralMoment( pState, 1, 1), CFitParamsCentralMomentError);
  *Uu11 = val;
  CHKCV( val = cvGetCentralMoment( pState, 2, 0), CFitParamsCentralMomentError);
  *Uu20 = val;
  CHKCV( val = cvGetCentralMoment( pState, 0, 2), CFitParamsCentralMomentError);
  *Uu02 = val;
  return CFitParamsNoError;
}
