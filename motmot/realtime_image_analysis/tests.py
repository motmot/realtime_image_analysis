import pkg_resources
import unittest
import motmot.realtime_image_analysis.realtime_image_analysis as ria
import motmot.realtime_image_analysis.slow as slow
import motmot.FastImage.FastImage as FastImage
import numpy

XT = 67
YT = 31

def result_validator_func(points):
    assert len(points)==1
    for pt in points:
        (x, y, area, slope, eccentricity) = pt
        if abs(x-XT) > 0.01:
            raise ValueError('x not close to %d'%XT)
        if abs(y-YT) > 0.01:
            raise ValueError('y not close to %d'%YT)
    return

def get_roi_im(w,h):
    sz = FastImage.Size(w,h)
    roi_im=FastImage.FastImage8u(sz)
    arr = numpy.asarray(roi_im) # numpy view
    arr[:,:] = 0
    arr[YT,XT] = 50
    return roi_im, result_validator_func

class TestRealtimeImageAnalysis(unittest.TestCase):
    def setUp(self):
        self.w = 640
        self.h = 480
        max_points = 3
        roi2_radius = 10
        lbrt = (0,0,self.w-1,self.h-1) # left, bottom, right, top
        self.ra = ria.RealtimeAnalyzer( lbrt, self.w,self.h, max_points, roi2_radius )

    def test_basic(self):
        timestamp = 0.0
        framenumber = 0
        use_roi2 = False
        use_cmp = False
        roi_im,result_validator_func=get_roi_im(self.w,self.h)
        points = self.ra.do_work(roi_im,timestamp,framenumber,use_roi2,use_cmp)
        result_validator_func( points )

    def test_basic_roi2(self):
        timestamp = 0.0
        framenumber = 0
        use_roi2 = True
        use_cmp = False
        roi_im,result_validator_func=get_roi_im(self.w,self.h)
        points = self.ra.do_work(roi_im,timestamp,framenumber,use_roi2,use_cmp)
        result_validator_func( points )

    def test_variance(self):
        timestamp = 0.0
        framenumber = 0
        use_roi2 = False
        use_cmp = True
        roi_im,result_validator_func=get_roi_im(self.w,self.h)
        points = self.ra.do_work(roi_im,timestamp,framenumber,use_roi2,use_cmp)
        result_validator_func( points )

    def test_variance_roi2(self):
        timestamp = 0.0
        framenumber = 0
        use_roi2 = True
        use_cmp = True
        roi_im,result_validator_func=get_roi_im(self.w,self.h)
        points = self.ra.do_work(roi_im,timestamp,framenumber,use_roi2,use_cmp)
        result_validator_func( points )

    def test_fast_vs_slow(self):
        h,w = 3,5
        shape = h,w # 3 rows, 5 cols

        results = []
        for func in [slow.do_bg_maint,
                     ria.do_bg_maint]:

            running_mean_im = 4*numpy.ones( shape, dtype=numpy.float32 )
            hw_roi_frame = 5*numpy.ones( shape, dtype=numpy.uint8 )
            max_frame_size = FastImage.Size( w,h )
            ALPHA = 0.25
            running_mean8u_im = numpy.empty( shape, dtype=numpy.uint8 )
            fastframef32_tmp = numpy.empty( shape, dtype=numpy.float32 )
            running_sumsqf = 16*numpy.ones( shape, dtype=numpy.float32 )
            mean2 = numpy.empty( shape, dtype=numpy.float32 )
            std2 = numpy.empty( shape, dtype=numpy.float32 )
            running_stdframe = numpy.empty( shape, dtype=numpy.float32 )
            n_sigma = 2.0
            compareframe8u = numpy.empty( shape, dtype=numpy.uint8 )
            bright_non_gaussian_cutoff = 255
            noisy_pixels_mask = numpy.ones( shape, dtype=numpy.uint8 )
            bright_non_gaussian_replacement = 5
            bench = 0

            func( FastImage.asfastimage(running_mean_im),
                  FastImage.asfastimage(hw_roi_frame),
                  max_frame_size,
                  ALPHA,
                  FastImage.asfastimage(running_mean8u_im),
                  FastImage.asfastimage(fastframef32_tmp),
                  FastImage.asfastimage(running_sumsqf),
                  FastImage.asfastimage(mean2),
                  FastImage.asfastimage(std2),
                  FastImage.asfastimage(running_stdframe),
                  n_sigma,
                  FastImage.asfastimage(compareframe8u),
                  bright_non_gaussian_cutoff,
                  FastImage.asfastimage(noisy_pixels_mask),
                  bright_non_gaussian_replacement,
                  bench=bench)

            results_order = ('running_mean8u_im',
                             'fastframef32_tmp',
                             'running_sumsqf',
                             'mean2',
                             'std2',
                             'running_stdframe',
                             )
            this_results = [ locals()[name] for name in results_order ]
            results.append( this_results )

        for i,(slow_result_arr, fast_result_arr) in enumerate(zip(*results)):
            name = results_order[i]
            if 0:
                print name
                print slow_result_arr
                print fast_result_arr
                print
            assert slow_result_arr.shape == fast_result_arr.shape
            assert numpy.allclose( slow_result_arr, fast_result_arr )

def get_test_suite():
    ts=unittest.TestSuite((unittest.makeSuite(TestRealtimeImageAnalysis),))
    return ts

if __name__ == '__main__':
    if 1:
        ts = get_test_suite()
        ts.debug()
    else:
        unittest.main()
