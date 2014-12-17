cdef extern from "opencv2/opencv.hpp" nogil:
     ctypedef struct CvMoments:
         int dummy # doesn't really exist
