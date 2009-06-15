import enthought.traits.api as traits
from enthought.traits.ui.api import View, Item, Group
import motmot.realtime_image_analysis.realtime_image_analysis as \
       realtime_image_analysis
import motmot.FastImage.FastImage as FastImage
import threading
import os,sys
import motmot.ufmf.ufmf as ufmf

class TraitedRealtimeAnalyzer(traits.HasTraits):
    """A traits-based wrapper around realtime_image_analysis.RealtimeAnalyzer"""
    max_width = traits.Int(None,transient=True) # XXX how to make set-once?
    max_height = traits.Int(None,transient=True) # XXX how to make set-once?
    pixel_format = traits.String(None,transient=True) # XXX how to make set-once?
    max_N_points = traits.Int(20)
    roi_radius = traits.Int(20)
    bg_Nth_frame = traits.Int(100)
    _ra = traits.Any(transient=True)
    do_work = traits.Any(transient=True)
    clear_and_take_BG = traits.Button()
    _clear_take_event = traits.Any(transient=True)
    #difference_threshold = traits.Float(5.0)
    use_roi2 = traits.Bool(True) # always true
    saving = traits.Bool(False)
    start_saving = traits.Button()
    stop_saving = traits.Button()

    traits_view = View(Group(Item(name='start_saving',show_label=False),
                             Item(name='stop_saving',show_label=False),
                             Item(name='max_N_points',style='readonly'), # temporary
                             Item(name='roi_radius',style='readonly'), # temporary
                             Item(name='bg_Nth_frame'),
                             Item(name='clear_and_take_BG',show_label=False),
                             )
                       )

    def __init__(self,*args,**kwargs):
        super(TraitedRealtimeAnalyzer,self).__init__(*args,**kwargs)
        if self.max_width is None:
            raise ValueError('max_width must be specified')
        if self.max_height is None:
            raise ValueError('max_height must be specified')
        if self.pixel_format is None:
            raise ValueError('pixel_format must be specified')
        lbrt = (0,0,self.max_width-1,self.max_height-1)
        self._ra = realtime_image_analysis.RealtimeAnalyzer(lbrt,
                                                            self.max_width,
                                                            self.max_height,
                                                            self.max_N_points,
                                                            self.roi_radius)
        # assign functions
        self.do_work = self._ra.do_work
        self.get_image_view = self._ra.get_image_view
        self._clear_take_event = threading.Event()
        self.ufmf = None

    def _start_saving_fired(self):
        print 'saving started'
        self.saving = True
        #self.ufmf_saver.start_saving()
        fname = 'test.ufmf'
        fname = os.path.abspath(fname)
        self.ufmf = ufmf.UfmfSaver(fname,
                                   max_width=self.max_width,
                                   max_height=self.max_height,
                                   coding=self.pixel_format,
                                   )
        print 'saving',fname

    def _stop_saving_fired(self):
        print 'saving stopped'
        self.saving = False
        #self.ufmf_saver.stop_saving()
        self.ufmf.close()
        self.ufmf = None

    def _clear_and_take_BG_fired(self):
        self._clear_take_event.set()

    def quit(self):
        if self.ufmf is not None:
            self.ufmf.close()
            self.ufmf = None

    def process_frame(self,buf,buf_offset,timestamp,framenumber):
        fibuf = FastImage.asfastimage(buf)
        l,b = buf_offset
        lbrt = l, b, l+fibuf.size.w-1, b+fibuf.size.h-1

        if self._clear_take_event.isSet():
            # reset the background image
            running_mean8u_im = realtime_analyzer.get_image_view('mean')
            if running_mean8u_im.size == fibuf.size:
                srcfi = fibuf
                bg_copy = srcfi.get_8u_copy(self.max_frame_size)
            else:
                srcfi = FastImage.FastImage8u(self.max_frame_size)
                srcfi_roi = srcfi.roi(l,b,fibuf.size)
                fibuf.get_8u_copy_put(srcfi_roi, fibuf.size)
                bg_copy = srcfi # newly created, no need to copy

            srcfi.get_32f_copy_put( self.running_mean_im,   self.max_frame_size )
            srcfi.get_8u_copy_put(  running_mean8u_im, self.max_frame_size )

            #self.ufmf_saver.put_image('bg',bg_copy)
            self.ufmf.add_keyframe('mean',bg_copy)
            self._clear_take_event.clear()
            del srcfi, bg_copy # don't pollute namespace

        xpoints = self.do_work( fibuf, timestamp, framenumber,self.use_roi2)

        if self.saving:
            ypoints = []
            w = h = self.roi_radius*2
            for pt in xpoints:
                ypoints.append( (pt[0],pt[1],w,h) )

            actual_saved_points = self.ufmf.add_frame( fibuf,
                                                       timestamp,
                                                       ypoints )
        else:
            actual_saved_points = []
        return xpoints, actual_saved_points

if __name__=='__main__':
    tra = TraitedRealtimeAnalyzer(max_width=640,max_height=480)
    tra.configure_traits()
