from ..builder import DETECTORS
from .two_stage_obb import TwoStageOBBDetector

@DETECTORS.register_module
class FasterRCNNOBB(TwoStageOBBDetector):

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 dump_folder=None,
                 neck=None,
                 pretrained=None):
        super(FasterRCNNOBB, self).__init__(
            dump_folder=dump_folder,
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
