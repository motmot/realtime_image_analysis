cimport fw

cdef extern from "c_fit_params.h":
    ctypedef enum CFitParamsReturnType:
        CFitParamsNoError
        CFitParamsZeroMomentError
        CFitParamsOtherError
        CFitParamsCentralMomentError
    ctypedef int cFitParamsMomentState_64f

    fw.FwStatus cFitParamsMomentInitAlloc_64f(cFitParamsMomentState_64f**)
    fw.FwStatus cFitParamsMomentFree_64f( cFitParamsMomentState_64f* )

    CFitParamsReturnType fit_params( cFitParamsMomentState_64f *pState,
                                     double *x0, double *y0,
                                     double *Mu00,
                                     double *Uu11, double *Uu20, double *Uu02,
                                     int width, int height, unsigned char *img,
                                     int img_step )
