#ifndef _RIA_MOMENTS_H
#define _RIA_MOMENTS_H
#include <fwBase.h>

typedef struct {
  void* private;
} cFitParamsMomentState_64f;

FwStatus cFitParamsMomentInitAlloc_64f(cFitParamsMomentState_64f**);
FwStatus cFitParamsMomentFree_64f( cFitParamsMomentState_64f* );
FwStatus print_state(cFitParamsMomentState_64f* state);

FwStatus cFitParamsMoments64f_8u_C1R( const Fw8u*, int, FwiSize, cFitParamsMomentState_64f* );
FwStatus cFitParamsGetSpatialMoment_64f( const cFitParamsMomentState_64f*, int, int, int, FwiPoint, Fw64f*);
FwStatus cFitParamsGetCentralMoment_64f( const cFitParamsMomentState_64f*, int, int, int, Fw64f*);

#endif /* _RIA_MOMENTS_H */
