#include <stdlib.h>
#include <stdio.h>
#include "moments.h"

typedef struct {
  int dummy;
} state_t;

FwStatus cFitParamsMomentInitAlloc_64f(cFitParamsMomentState_64f** state_ptr) {
  state_t* internal;

  internal = malloc(sizeof(state_t));
  if (internal==NULL) {
    return fwStsMemAllocErr;
  }

  internal->dummy = 1234;

  *state_ptr = malloc( sizeof(cFitParamsMomentState_64f) );
  if (*state_ptr == NULL) {
    return fwStsMemAllocErr;
  }
  (*state_ptr)->private = internal;
  return fwStsNoErr;
}

#define _get_internal()                         \
  state_t* internal;                            \
  if(state==NULL) {                             \
    return fwStsNullPtrErr;                     \
  }                                             \
  internal=state->private;                      \
  if(internal==NULL) {                          \
    return fwStsNullPtrErr;                     \
  }

FwStatus cFitParamsMomentFree_64f( cFitParamsMomentState_64f* state) {
  _get_internal();
  free(internal);
  free(state);
  return fwStsNoErr;
}

FwStatus print_state(cFitParamsMomentState_64f* state) {
  _get_internal();
  printf("state = %d\n",internal->dummy);
  return fwStsNoErr;
}

#define NOT_IMPLEMENTED if(1) {return fwStsMemAllocErr;}

FwStatus cFitParamsMoments64f_8u_C1R( const Fw8u* src, int step, FwiSize size,
                                      cFitParamsMomentState_64f* state ) {
  _get_internal();
  NOT_IMPLEMENTED;
  return fwStsNoErr;
}

FwStatus cFitParamsGetSpatialMoment_64f( const cFitParamsMomentState_64f* state,
                                         int p, int q, int chan,
                                         FwiPoint offset,
                                         Fw64f* dest) {
  _get_internal();
  NOT_IMPLEMENTED;
  return fwStsNoErr;
}

FwStatus cFitParamsGetCentralMoment_64f( const cFitParamsMomentState_64f* state,
                                         int p, int q, int chan,
                                         Fw64f* dest) {
  _get_internal();
  NOT_IMPLEMENTED;
  return fwStsNoErr;
}

