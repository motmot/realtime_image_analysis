import unittest
import motmot.realtime_image_analysis.realtime_image_analysis as ria
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

def get_test_suite():
    ts=unittest.TestSuite((unittest.makeSuite(TestRealtimeImageAnalysis),))
    return ts

if __name__ == '__main__':
    unittest.main()
