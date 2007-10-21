import unittest
import realtime_image_analysis as ria
import FastImage
import numpy

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
        sz = FastImage.Size(self.w,self.h)
        roi_im=FastImage.FastImage8u(sz)
        self.ra.do_work(roi_im,timestamp,framenumber,use_roi2,use_cmp)
        
    def test_basic_roi2(self):
        timestamp = 0.0
        framenumber = 0
        use_roi2 = True
        use_cmp = False
        sz = FastImage.Size(self.w,self.h)
        roi_im=FastImage.FastImage8u(sz)
        self.ra.do_work(roi_im,timestamp,framenumber,use_roi2,use_cmp)

    def test_variance(self):
        timestamp = 0.0
        framenumber = 0
        use_roi2 = False
        use_cmp = True
        sz = FastImage.Size(self.w,self.h)
        roi_im=FastImage.FastImage8u(sz)
        self.ra.do_work(roi_im,timestamp,framenumber,use_roi2,use_cmp)
        
    def test_variance_roi2(self):
        timestamp = 0.0
        framenumber = 0
        use_roi2 = True
        use_cmp = True
        sz = FastImage.Size(self.w,self.h)
        roi_im=FastImage.FastImage8u(sz)
        self.ra.do_work(roi_im,timestamp,framenumber,use_roi2,use_cmp)

def get_test_suite():
    ts=unittest.TestSuite((unittest.makeSuite(TestRealtimeImageAnalysis),))
    return ts

if __name__ == '__main__':
    unittest.main()
