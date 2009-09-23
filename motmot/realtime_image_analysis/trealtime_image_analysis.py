import enthought.traits.api as traits
from enthought.traits.ui.api import View, Item, Group
import motmot.realtime_image_analysis.realtime_image_analysis as \
       realtime_image_analysis

class TraitedRealtimeAnalyzer(traits.HasTraits):
    """A traits-based wrapper around realtime_image_analysis.RealtimeAnalyzer"""
    max_width = traits.Int(None,transient=True) # XXX how to make set-once?
    max_height = traits.Int(None,transient=True) # XXX how to make set-once?
    max_N_points = traits.Int(10)
    roi2_radius = traits.Int(20)
    _ra = traits.Any(transient=True)
    do_work = traits.Any(transient=True)

    traits_view = View(Group(Item(name='max_N_points'),
                             Item(name='roi2_radius'),
                             )
                       )

    def __init__(self,*args,**kwargs):
        super(TraitedRealtimeAnalyzer,self).__init__(*args,**kwargs)
        if self.max_width is None:
            raise ValueError('max_width must be specified')
        if self.max_height is None:
            raise ValueError('max_height must be specified')
        lbrt = (0,0,self.max_width-1,self.max_height-1)
        self._ra = realtime_image_analysis.RealtimeAnalyzer(lbrt,
                                                            self.max_width,
                                                            self.max_height,
                                                            self.max_N_points,
                                                            self.roi2_radius)
        # assign functions
        self.do_work = self._ra.do_work
        self.get_image_view = self._ra.get_image_view

if __name__=='__main__':
    tra = TraitedRealtimeAnalyzer(max_width=640,max_height=480)
    tra.configure_traits()
