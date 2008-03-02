# emacs, this is -*-Python-*- mode

# This code serves as the basis for realtime trackers.  The basic idea
# is to use the Intel IPP Library to do the heavy lifting and release
# Python's global interpreter lock (GIL) whenever possible, so that
# work can continue in other threads. This allows, for example,
# processing images on one CPU while grabbing them on another.

import time
import numpy as nx
import warnings

#cimport FastImage
cimport motmot.FastImage.FastImage as FastImage
import motmot.FastImage.FastImage as FastImage

cdef double nan
nan = nx.nan

cimport c_lib
cimport c_python

cimport ipp

cdef extern from "unistd.h":
    ctypedef long intptr_t

cdef extern from "c_fit_params.h":
    ctypedef enum CFitParamsReturnType:
        CFitParamsNoError
        CFitParamsZeroMomentError
        CFitParamsOtherError
        CFitParamsCentralMomentError
    CFitParamsReturnType fit_params( ipp.IppiMomentState_64f *pState, double *x0, double *y0,
                                     double *Mu00,
                                     double *Uu11, double *Uu20, double *Uu02,
                                     int width, int height, unsigned char *img, int img_step )

cdef extern from "eigen.h":
    int eigen_2x2_real( double A, double B, double C, double D,
                        double *evalA, double *evecA1,
                        double *evalB, double *evecB1 )

cdef extern from "motmot_ipp_macros.h":
    ipp.Ipp8u*  IMPOS8u(  ipp.Ipp8u*  im, int step, int bottom, int left)
    ipp.Ipp32f* IMPOS32f( ipp.Ipp32f* im, int step, int bottom, int left)
    void CHK_NOGIL( ipp.IppStatus errval )
    void SET_ERR( int errval )

cdef extern from "c_time_time.h":
    double time_time()

cdef void CHK_HAVEGIL( ipp.IppStatus errval ) except *:
    if (errval != ipp.ippStsNoErr):
        raise FastImage.IppError(errval)

##cdef void SET_ERR( int errval ):
##    # This is rather crude at the moment because calls to the Python C
##    # API cannot be made.  (This code is executed when we have
##    # released the GIL.)
##    c_lib.printf("SET_ERR(%d) called! (May not have GIL, cannot raise exception.)\n",errval)
##    c_lib.exit(2)

cdef print_8u_arr(ipp.Ipp8u* src,int width,int height,int src_step):
  cdef int row, col
  cdef ipp.Ipp8u* src_rowstart
  for row from 0 <= row < height:
    src_rowstart = src+(row*src_step);
    for col from 0 <= col < width:
      print "%d"%src_rowstart[col],
    print
  print

cdef class RealtimeAnalyzer:

    # full image size
    cdef int maxwidth, maxheight

    # ROI size
    cdef int _left, _bottom, _right, _top

    cdef int _roi2_radius

    # runtime parameters
    cdef ipp.Ipp8u _diff_threshold
    cdef float _clear_threshold

    cdef ipp.Ipp8u _despeckle_threshold

    cdef FastImage.Size _roi_sz

    cdef int n_rot_samples

    # class A images:

    # Portions of these images may be out of date. This is done
    # because we only want to malloc each image once, despite changing
    # input image size. The roi is a view of the active part. The roi2
    # is a view of the sub-region of the active part.

    cdef FastImage.FastImage8u absdiff_im, cmpdiff_im # also raw_im

    cdef FastImage.FastImage8u absdiff_im_roi_view
    cdef FastImage.FastImage8u cmpdiff_im_roi_view
    cdef FastImage.FastImage8u absdiff_im_roi2_view
    cdef FastImage.FastImage8u cmpdiff_im_roi2_view

    # class B images:

    # These entirety of these images are always valid and have an roi
    # into the active region.

    cdef FastImage.FastImage8u mask_im, mean_im, cmp_im

    cdef FastImage.FastImage8u mean_im_roi_view, cmp_im_roi_view

    # other stuff
    cdef ipp.IppiMomentState_64f *pState

    cdef object imname2im
    cdef int max_num_points

    def __new__(self,*args,**kw):
        # image moment calculation initialization
        CHK_HAVEGIL( ipp.ippiMomentInitAlloc_64f( &self.pState, ipp.ippAlgHintFast ) )

    def __dealloc__(self):
        CHK_HAVEGIL( ipp.ippiMomentFree_64f( self.pState ))

    def __init__(self,object lbrt,int maxwidth, int maxheight, int max_num_points, int roi2_radius):
        # software analysis ROI
        self.maxwidth = maxwidth
        self.maxheight = maxheight

        self.max_num_points = max_num_points
        #print 'realtime analysis with %d points maximum starting'%self.max_num_points

        self._roi2_radius = roi2_radius
        self._diff_threshold = 11
        self._clear_threshold = 0.2

        self._despeckle_threshold = 5
        self.n_rot_samples = 100*60 # 1 minute

        sz = FastImage.Size(self.maxwidth,self.maxheight)
        # 8u images
        self.absdiff_im=FastImage.FastImage8u(sz)

        # 8u background
        self.mask_im=FastImage.FastImage8u(sz)
        self.mask_im.set_val(0,sz)

        self.mean_im=FastImage.FastImage8u(sz)
        self.cmp_im=FastImage.FastImage8u(sz)
        self.cmpdiff_im=FastImage.FastImage8u(sz)

        # create and update self.imname2im dict
        self.imname2im = {'absdiff' :self.absdiff_im,
                          'mean'    :self.mean_im,
                          'mask'    :self.mask_im,
                          'cmp'     :self.cmp_im,
                          'cmpdiff' :self.cmpdiff_im,
                          }

        self.roi = lbrt

        # initialize background images
        self.mean_im_roi_view.set_val( 0, self._roi_sz )

    def distort( self, float x0, float y0 ):
        if self._helper is None:
            return x0, y0
        return self._helper.distort( x0, y0 )

    def do_work(self,
                FastImage.FastImage8u raw_im_small,
                double timestamp,
                int framenumber,
                int use_roi2,
                int use_cmp=0,
                double max_duration_sec=0.0
                ):
        """find location and orientation of local points (fast enough for realtime use)

        inputs
        ------

        timestamp
        framenumber
        use_roi2

        optional inputs (default to 0)
        ------------------------------
        use_cmp -- perform more detailed analysis against cmp image (used with ongoing variance estimation)

        outputs
        -------

        [ (x0_abs, y0_abs, area, slope, eccentricity) ] -- with a tuple for each point found

        """
        cdef double x0, y0
        cdef double x0_abs, y0_abs, area
        cdef double rise, run, slope, eccentricity
        cdef double evalA, evalB
        cdef double evecA1, evecB1

        cdef double Mu00, Uu11, Uu02, Uu20
        cdef int i
        cdef int result, eigen_err

        cdef int index_x,index_y

        cdef ipp.Ipp8u max_val
        cdef ipp.Ipp8u* max_val_ptr
        cdef ipp.Ipp8u max_std_diff

        cdef ipp.Ipp8u clear_despeckle_thresh

        cdef FastImage.Size roi2_sz
        cdef int left2, right2, bottom2, top2

        cdef int found_point
        cdef int n_found_points

        cdef double entry_time
        cdef double now

        entry_time = time.time()

        roi2_sz = FastImage.Size(21,21)

        all_points_found = [] # len(all_points_found) == n_found_points
        n_found_points = 0

        # This is our near-hack to ensure users update .roi before calling .do_work()
        if not ((self._roi_sz.sz.width == raw_im_small.imsiz.sz.width) and
                (self._roi_sz.sz.height == raw_im_small.imsiz.sz.height)):
            raise ValueError("input image size (%s) does not correspond to ROI (%s)"
                             "(set RealtimeAnalyzer.roi before calling)"%(
                str(raw_im_small.imsiz),str(self._roi_sz)))

        # find difference from mean
        c_python.Py_BEGIN_ALLOW_THREADS
        # absdiff_im = | raw_im - mean_im |
        #raw_im_small.fast_get_absdiff_put( self.mean_im_roi_view,
        #                                   self.absdiff_im_roi_view,
        #                                   self._roi_sz) # does own CHK
        CHK_NOGIL( ipp.ippiAbsDiff_8u_C1R(<ipp.Ipp8u*>self.mean_im_roi_view.im,self.mean_im_roi_view.step,
                                          <ipp.Ipp8u*>raw_im_small.im,raw_im_small.step,
                                          <ipp.Ipp8u*>self.absdiff_im_roi_view.im,self.absdiff_im_roi_view.step,
                                          self._roi_sz.sz))

        # mask unused part of absdiff_im to 0
        #self.absdiff_im.fast_set_val_masked( 0,
        #                                     self.mask_im,
        #                                     self.absdiff_im.imsiz) # does own CHK

        CHK_NOGIL( ipp.ippiSet_8u_C1MR( 0, <ipp.Ipp8u*>self.absdiff_im.im, self.absdiff_im.step, self.absdiff_im.imsiz.sz,
                                        <ipp.Ipp8u*>self.mask_im.im, self.mask_im.step))

        if use_cmp:
            # clip the minimum comparison value to diff_threshold
            CHK_NOGIL( ipp.ippiThreshold_Val_8u_C1IR(<ipp.Ipp8u*>self.cmp_im_roi_view.im,
                                               self.cmp_im_roi_view.step,
                                               self._roi_sz.sz,
                                               self._diff_threshold, self._diff_threshold, ipp.ippCmpLess))
        c_python.Py_END_ALLOW_THREADS

        while n_found_points < self.max_num_points:
            if max_duration_sec != 0.0:
                now = time.time()
                if (now-entry_time) > max_duration_sec:
                    warnings.warn('stopped looking for points (this frame only) because it was taking longer than max_duration_sec',
                                  stacklevel=1)
                    break
            # release GIL
            c_python.Py_BEGIN_ALLOW_THREADS

            # WARNING WARNING WARNING WARNING WARNING WARNING WARNING

            # Everything from here to Py_END_ALLOW_THREADS must not make
            # calls to the Python C API.  The Python GIL (Global
            # Interpreter Lock) has been released, meaning that any calls
            # to the Python interpreter will have undefined effects,
            # because the interpreter is presumably in the middle of
            # another thread right now.

            # If you are not sure whether or not calls use the Python C
            # API, check the .c file generated by Pyrex.  Make sure even
            # function calls do not call the Python C API.

            # find max pixel in ROI
            if use_cmp:
                # cmpdiff_im = absdiff_im - cmp_im (saturates 8u)
                #self.absdiff_im_roi_view.fast_get_sub_put( self.cmp_im_roi_view,
                #                                           self.cmpdiff_im_roi_view,
                #                                           self._roi_sz ) # down own CHK
                CHK_NOGIL( ipp.ippiSub_8u_C1RSfs(<ipp.Ipp8u*>self.cmp_im_roi_view.im, self.cmp_im_roi_view.step,
                                                 <ipp.Ipp8u*>self.absdiff_im_roi_view.im, self.absdiff_im_roi_view.step,
                                                 <ipp.Ipp8u*>self.cmpdiff_im_roi_view.im, self.cmpdiff_im_roi_view.step,
                                                 self._roi_sz.sz,0))
                CHK_NOGIL( ipp.ippiMaxIndx_8u_C1R(
                    <ipp.Ipp8u*>self.cmpdiff_im_roi_view.im,self.cmpdiff_im_roi_view.step,
                    self._roi_sz.sz, &max_std_diff, &index_x, &index_y))
                max_val_ptr = (<ipp.Ipp8u*>self.absdiff_im_roi_view.im)+self.absdiff_im_roi_view.step*index_y+index_x
                max_val = max_val_ptr[0] # value at maximum difference from std
            else:
                CHK_NOGIL( ipp.ippiMaxIndx_8u_C1R(
                    <ipp.Ipp8u*>self.absdiff_im_roi_view.im,self.absdiff_im_roi_view.step,
                    self._roi_sz.sz, &max_val, &index_x, &index_y))

            if use_roi2:
                # find mini-ROI for further analysis (defined in non-ROI space)
                left2 = index_x - self._roi2_radius + self._left
                right2 = index_x + self._roi2_radius + self._left
                bottom2 = index_y - self._roi2_radius + self._bottom
                top2 = index_y + self._roi2_radius + self._bottom

                if left2 < self._left: left2 = self._left
                if right2 > self._right: right2 = self._right
                if bottom2 < self._bottom: bottom2 = self._bottom
                if top2 > self._top: top2 = self._top
                roi2_sz.sz.width = right2 - left2 + 1
                roi2_sz.sz.height = top2 - bottom2 + 1
            else:
                left2 = self._left
                right2 = self._right
                bottom2 = self._bottom
                top2 = self._top
                roi2_sz.sz.width = self._roi_sz.sz.width
                roi2_sz.sz.height = self._roi_sz.sz.height

            c_python.Py_END_ALLOW_THREADS
            # XXX should figure out how not to release GIL here...
            self.absdiff_im_roi2_view = self.absdiff_im.roi(left2,bottom2,roi2_sz)
            if use_cmp:
                self.cmpdiff_im_roi2_view = self.cmpdiff_im.roi(left2,bottom2,roi2_sz)
            c_python.Py_BEGIN_ALLOW_THREADS

            # (to reduce moment arm:) if pixel < self._clear_threshold*max(pixel): pixel=0

            clear_despeckle_thresh = <ipp.Ipp8u>(self._clear_threshold*max_val)
            if clear_despeckle_thresh < self._despeckle_threshold:
                clear_despeckle_thresh = self._despeckle_threshold

            CHK_NOGIL( ipp.ippiThreshold_Val_8u_C1IR(
                <ipp.Ipp8u*>self.absdiff_im_roi2_view.im,self.absdiff_im_roi2_view.step,
                roi2_sz.sz, clear_despeckle_thresh, 0, ipp.ippCmpLess))

            found_point = 1

            if not use_cmp:
                if max_val < self._diff_threshold:
                    x0=nan
                    y0=nan
                    x0_abs = nan
                    y0_abs = nan
                    found_point = 0 # c int (bool)
                    max_val = 0
            else:
                if max_std_diff == 0:
                    x0=nan
                    y0=nan
                    x0_abs = nan
                    y0_abs = nan
                    found_point = 0 # c int (bool)
            if found_point:
                result = fit_params( self.pState, &x0, &y0,
                                     &Mu00,
                                     &Uu11, &Uu20, &Uu02,
                                     roi2_sz.sz.width, roi2_sz.sz.height,
                                     <unsigned char*>self.absdiff_im_roi2_view.im,
                                     self.absdiff_im_roi2_view.step)
                # note that x0 and y0 are now relative to the ROI origin
                if result == CFitParamsNoError:
                    area = Mu00
                    eigen_err = eigen_2x2_real( Uu20, Uu11,
                                                Uu11, Uu02,
                                                &evalA, &evecA1,
                                                &evalB, &evecB1)
                    if eigen_err:
                        slope = nan
                        eccentricity = 0.0
                    else:
                        rise = 1.0 # 2nd component of eigenvectors will always be 1.0
                        if evalA > evalB:
                            run = evecA1
                            eccentricity = evalA/evalB
                        else:
                            run = evecB1
                            eccentricity = evalB/evalA
                        slope = rise/run

                elif result == CFitParamsZeroMomentError:
                    x0 = nan
                    y0 = nan
                    x0_abs = nan
                    y0_abs = nan
                    found_point = 0
                elif result == CFitParamsCentralMomentError:
                    slope = nan
                else: SET_ERR(1)

                # set x0 and y0 relative to whole frame
                if found_point:
                    x0_abs = x0+left2
                    y0_abs = y0+bottom2

            # grab GIL
            c_python.Py_END_ALLOW_THREADS

            if found_point:
                if not (n_found_points+1==self.max_num_points): # only modify the image if more follow...
                    self.absdiff_im_roi2_view.set_val(0, roi2_sz )

            if not found_point:
                break

            all_points_found.append(
                (x0_abs, y0_abs, area, slope, eccentricity)
                )
            n_found_points = n_found_points+1

        return all_points_found

    def get_image_view(self,which='mean'):
        im = self.imname2im[which]
        return im

    property roi2_radius:
        def __get__(self):
            return self._roi2_radius
        def __set__(self,value):
            self._roi2_radius = value

    property clear_threshold:
        def __get__(self):
            return self._clear_threshold
        def __set__(self,value):
            self._clear_threshold = value

    property diff_threshold:
        def __get__(self):
            return self._diff_threshold
        def __set__(self,value):
            self._diff_threshold = value

    property scale_factor:
        def __get__(self):
            return self._scale_factor
        def __set__(self,value):
            print 'setting scale_factor to',value
            self._scale_factor = value

    property roi:
        def __get__(self):
            return (self._left,self._bottom,self._right,self._top)
        def __set__(self,object lbrt):
            cdef int l,b,r,t

            l,b,r,t = lbrt
            if (l==self._left and b==self._bottom and
                r==self._right and t==self._top):
                # nothing to do
                return

            self._left = l
            self._bottom = b
            self._right = r
            self._top = t

            if self._left < 0:
                raise ValueError('attempting to set ROI left %d'%(self._left))
            if self._bottom < 0:
                raise ValueError('attempting to set ROI bottom %d'%(self._bottom))
            if self._right >= self.maxwidth:
                raise ValueError('attempting to set ROI right to %d (width %d)'%(self._right,self.maxwidth))
            if self._top >= self.maxheight:
                raise ValueError('attempting to set ROI top to %d (height %d)'%(self._top,self.maxheight))

            self._roi_sz = FastImage.Size( self._right-self._left+1, self._top-self._bottom+1 )

            self.absdiff_im_roi_view = self.absdiff_im.roi(self._left,self._bottom,self._roi_sz)
            self.mean_im_roi_view = self.mean_im.roi(self._left,self._bottom,self._roi_sz)
            self.cmp_im_roi_view = self.cmp_im.roi(self._left,self._bottom,self._roi_sz)
            self.cmpdiff_im_roi_view = self.cmpdiff_im.roi(self._left,self._bottom,self._roi_sz)

def bg_help_slow( FastImage.FastImage32f running_mean_im,
                  FastImage.FastImage32f fastframef32_tmp,
                  FastImage.FastImage32f running_sumsqf,
                  FastImage.FastImage32f mean2,
                  FastImage.FastImage32f running_stdframe,
                  FastImage.FastImage8u running_mean8u_im,
                  FastImage.FastImage8u hw_roi_frame,
                  FastImage.FastImage8u noisy_pixels_mask,
                  FastImage.FastImage8u compareframe8u,
                  FastImage.Size max_frame_size,
                  double ALPHA,
                  double C):
    # maintain running average
    running_mean_im.toself_add_weighted( hw_roi_frame, max_frame_size, ALPHA )
    # maintain 8bit unsigned background image
    running_mean_im.get_8u_copy_put( running_mean8u_im, max_frame_size )

    # standard deviation calculation
    hw_roi_frame.get_32f_copy_put(fastframef32_tmp,max_frame_size)
    fastframef32_tmp.toself_square(max_frame_size) # current**2
    running_sumsqf.toself_add_weighted( fastframef32_tmp, max_frame_size, ALPHA)
    running_mean_im.get_square_put(mean2,max_frame_size)
    running_sumsqf.get_subtracted_put(mean2,running_stdframe,max_frame_size)

    # now create frame for comparison
    C = 6.0
    running_stdframe.toself_multiply(C,max_frame_size)
    running_stdframe.get_8u_copy_put(compareframe8u,max_frame_size)

    # now we do hack, erm, heuristic for bright points, which aren't gaussian.
    running_mean8u_im.get_compare_put( 200, noisy_pixels_mask, max_frame_size, FastImage.CmpGreater)
    compareframe8u.set_val_masked(25, noisy_pixels_mask, max_frame_size)


def bg_help( FastImage.FastImage32f running_mean_im,
             FastImage.FastImage32f fastframef32_tmp,
             FastImage.FastImage32f running_sumsqf,
             FastImage.FastImage32f mean2,
             FastImage.FastImage32f running_stdframe,
             FastImage.FastImage8u running_mean8u_im,
             FastImage.FastImage8u hw_roi_frame,
             FastImage.FastImage8u noisy_pixels_mask,
             FastImage.FastImage8u compareframe8u,
             FastImage.Size max_frame_size,
             float ALPHA,
             double C):

    c_python.Py_BEGIN_ALLOW_THREADS # release GIL
    # maintain running average
    #running_mean_im.fast_toself_add_weighted_8u( hw_roi_frame, max_frame_size, ALPHA ) # done
    CHK_NOGIL( ipp.ippiAddWeighted_8u32f_C1IR( <ipp.Ipp8u*>hw_roi_frame.im, hw_roi_frame.step,
                                               <ipp.Ipp32f*>running_mean_im.im, running_mean_im.step,
                                               max_frame_size.sz, ALPHA))

    # maintain 8bit unsigned background image
    #running_mean_im.fast_get_8u_copy_put( running_mean8u_im, max_frame_size ) # done
    CHK_NOGIL( ipp.ippiConvert_32f8u_C1R(
        <ipp.Ipp32f*>running_mean_im.im,running_mean_im.step,
        <ipp.Ipp8u*>running_mean8u_im.im,running_mean8u_im.step,
        max_frame_size.sz, ipp.ippRndNear ))

    # standard deviation calculation
    #hw_roi_frame.fast_get_32f_copy_put(fastframef32_tmp,max_frame_size) # done
    CHK_NOGIL( ipp.ippiConvert_8u32f_C1R(<ipp.Ipp8u*>hw_roi_frame.im, hw_roi_frame.step,
                                         <ipp.Ipp32f*>fastframef32_tmp.im, fastframef32_tmp.step,
                                         max_frame_size.sz ))

    #fastframef32_tmp.fast_toself_square(max_frame_size) # current**2 # done
    CHK_NOGIL( ipp.ippiSqr_32f_C1IR( <ipp.Ipp32f*>fastframef32_tmp.im, fastframef32_tmp.step,
                                     max_frame_size.sz))

    #running_sumsqf.fast_toself_add_weighted_32f( fastframef32_tmp, max_frame_size, ALPHA) # done
    CHK_NOGIL( ipp.ippiAddWeighted_32f_C1IR( <ipp.Ipp32f*>fastframef32_tmp.im, fastframef32_tmp.step,
                                             <ipp.Ipp32f*>running_sumsqf.im, running_sumsqf.step,
                                             max_frame_size.sz, ALPHA))
    #running_mean_im.fast_get_square_put(mean2,max_frame_size) # done
    CHK_NOGIL( ipp.ippiSqr_32f_C1R( <ipp.Ipp32f*>running_mean_im.im, running_mean_im.step,
                                    <ipp.Ipp32f*>mean2.im, mean2.step,
                                    max_frame_size.sz))
    #running_sumsqf.fast_get_subtracted_put(mean2,running_stdframe,max_frame_size) # done
    CHK_NOGIL( ipp.ippiSub_32f_C1R(<ipp.Ipp32f*>mean2.im, mean2.step,
                                   <ipp.Ipp32f*>running_sumsqf.im, running_sumsqf.step,
                                   <ipp.Ipp32f*>running_stdframe.im, running_stdframe.step,
                                   max_frame_size.sz))

    # now create frame for comparison
    #running_stdframe.fast_toself_multiply(C,max_frame_size) # done
    CHK_NOGIL( ipp.ippiMulC_32f_C1IR(C, <ipp.Ipp32f*>running_stdframe.im, running_stdframe.step, max_frame_size.sz))

    #running_stdframe.fast_get_8u_copy_put(compareframe8u,max_frame_size) # done
    CHK_NOGIL( ipp.ippiConvert_32f8u_C1R(
        <ipp.Ipp32f*>running_stdframe.im,running_stdframe.step,
        <ipp.Ipp8u*>compareframe8u.im,compareframe8u.step,
        max_frame_size.sz, ipp.ippRndNear ))

    # now we do hack, erm, heuristic for bright points, which aren't gaussian.
    #running_mean8u_im.fast_get_compare_int_put_greater( 200, noisy_pixels_mask, max_frame_size)
    CHK_NOGIL( ipp.ippiCompareC_8u_C1R( <ipp.Ipp8u*>running_mean8u_im.im,running_mean8u_im.step,
                                        200,
                                        <ipp.Ipp8u*>noisy_pixels_mask.im,noisy_pixels_mask.step,
                                        max_frame_size.sz,
                                        ipp.ippCmpGreater))

    #compareframe8u.fast_set_val_masked(25, noisy_pixels_mask, max_frame_size) # done
    CHK_NOGIL( ipp.ippiSet_8u_C1MR( 25, <ipp.Ipp8u*>compareframe8u.im, compareframe8u.step, max_frame_size.sz,
                                    <ipp.Ipp8u*>noisy_pixels_mask.im, noisy_pixels_mask.step))
    c_python.Py_END_ALLOW_THREADS # acquire GIL

def do_bg_maint( FastImage.FastImage32f running_mean_im,
                 FastImage.FastImage8u hw_roi_frame,
                 FastImage.Size max_frame_size,
                 float ALPHA,
                 FastImage.FastImage8u running_mean8u_im,
                 FastImage.FastImage32f fastframef32_tmp,
                 FastImage.FastImage32f running_sumsqf,
                 FastImage.FastImage32f mean2,
                 FastImage.FastImage32f running_stdframe,
                 float n_sigma,
                 FastImage.FastImage8u compareframe8u,
                 int bright_non_gaussian_cutoff,
                 FastImage.FastImage8u noisy_pixels_mask,
                 int bright_non_gaussian_replacement,
                 int bench=0):
    cdef int BENCHMARK
    cdef int RAW_IPP
    BENCHMARK = bench
    cdef double t41, t42, t43, t44, t45, t46, t47, t48, t49, t491, t492

    c_python.Py_BEGIN_ALLOW_THREADS

    # maintain running average
    CHK_NOGIL( ipp.ippiAddWeighted_8u32f_C1IR( <ipp.Ipp8u*>hw_roi_frame.im, hw_roi_frame.step,
                                               <ipp.Ipp32f*>running_mean_im.im, running_mean_im.step,
                                               max_frame_size.sz, ALPHA))
    if BENCHMARK:
        t41 = time_time()

    CHK_NOGIL( ipp.ippiConvert_32f8u_C1R(
        <ipp.Ipp32f*>running_mean_im.im,running_mean_im.step,
        <ipp.Ipp8u*>running_mean8u_im.im,running_mean8u_im.step,
        max_frame_size.sz, ipp.ippRndNear ))

    if BENCHMARK:
        t42 = time_time()

    # standard deviation calculation
    CHK_NOGIL( ipp.ippiConvert_8u32f_C1R(<ipp.Ipp8u*>hw_roi_frame.im, hw_roi_frame.step,
                                         <ipp.Ipp32f*>fastframef32_tmp.im, fastframef32_tmp.step,
                                         max_frame_size.sz ))
    if BENCHMARK:
        t43 = time_time()

    CHK_NOGIL( ipp.ippiSqr_32f_C1IR( <ipp.Ipp32f*>fastframef32_tmp.im, fastframef32_tmp.step,
                                     max_frame_size.sz))
    if BENCHMARK:
        t44 = time_time()

    CHK_NOGIL( ipp.ippiAddWeighted_32f_C1IR( <ipp.Ipp32f*>fastframef32_tmp.im, fastframef32_tmp.step,
                                             <ipp.Ipp32f*>running_sumsqf.im, running_sumsqf.step,
                                             max_frame_size.sz, ALPHA))
    if BENCHMARK:
        t45 = time_time()

    ### GETS SLOWER
    CHK_NOGIL( ipp.ippiSqr_32f_C1R( <ipp.Ipp32f*>running_mean_im.im, running_mean_im.step,
                                    <ipp.Ipp32f*>mean2.im, mean2.step,
                                    max_frame_size.sz))
    if BENCHMARK:
        t46 = time_time()

    CHK_NOGIL( ipp.ippiSub_32f_C1R(<ipp.Ipp32f*>mean2.im, mean2.step,
                                   <ipp.Ipp32f*>running_sumsqf.im, running_sumsqf.step,
                                   <ipp.Ipp32f*>running_stdframe.im, running_stdframe.step,
                                   max_frame_size.sz))
    if BENCHMARK:
        t47 = time_time()

    # now create frame for comparison
    CHK_NOGIL( ipp.ippiMulC_32f_C1IR(n_sigma, <ipp.Ipp32f*>running_stdframe.im, running_stdframe.step, max_frame_size.sz))

    if BENCHMARK:
        t48 = time_time()

    CHK_NOGIL( ipp.ippiConvert_32f8u_C1R(
        <ipp.Ipp32f*>running_stdframe.im,running_stdframe.step,
        <ipp.Ipp8u*>compareframe8u.im,compareframe8u.step,
        max_frame_size.sz, ipp.ippRndNear ))

    if BENCHMARK:
        t49 = time_time()

    # now we do hack, erm, heuristic for bright points, which aren't gaussian.
    CHK_NOGIL( ipp.ippiCompareC_8u_C1R( <ipp.Ipp8u*>running_mean8u_im.im,running_mean8u_im.step,
                                        bright_non_gaussian_cutoff,
                                        <ipp.Ipp8u*>noisy_pixels_mask.im,noisy_pixels_mask.step,
                                        max_frame_size.sz,ipp.ippCmpGreater))

    if BENCHMARK:
        t491 = time_time()

    CHK_NOGIL( ipp.ippiSet_8u_C1MR( bright_non_gaussian_replacement,
                                    <ipp.Ipp8u*>compareframe8u.im, compareframe8u.step, max_frame_size.sz,
                                    <ipp.Ipp8u*>noisy_pixels_mask.im, noisy_pixels_mask.step))
    if BENCHMARK:
        t492 = time_time()

    c_python.Py_END_ALLOW_THREADS

    if BENCHMARK:
        res = t41, t42, t43, t44, t45, t46, t47, t48, t49, t491, t492
    else:
        res = None
    return res


