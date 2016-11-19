cimport cython_ipp.ipp as ipp

cdef extern from "c_fit_params.h" nogil:
    ctypedef enum CFitParamsReturnType:
        CFitParamsNoError
        CFitParamsZeroMomentError
        CFitParamsOtherError
        CFitParamsCentralMomentError
    CFitParamsReturnType fit_params( ipp.IppiMomentState_64f *pState,
                                     double *x0, double *y0,
                                     double *Mu00,
                                     double *Uu11, double *Uu20, double *Uu02,
                                     int width, int height, unsigned char *img,
                                     int img_step )
