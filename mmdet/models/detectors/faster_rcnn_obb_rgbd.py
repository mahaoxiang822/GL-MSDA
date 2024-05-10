from .two_stage_obb import TwoStageOBBDetector
import torch
import torch.nn as nn

from mmdet.core import (bbox2result, bbox2roi, rbbox_to_hbbox_list,
                        points_to_xywhtheta_list,  xywhtheta_to_rect_grasp_group)
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
from mmcv.runner import auto_fp16
import os


@DETECTORS.register_module
class FasterRCNNOBBRGBD(TwoStageOBBDetector):

    def __init__(self,
                 dump_folder,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None):
        super(FasterRCNNOBBRGBD, self).__init__(
            dump_folder=dump_folder,
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)

    def generate_rgd(self, rgb, depth):
        rgds = []
        for i in range(len(rgb)):
            rg = rgb[i, :2, :, :]
            d = depth[i]
            rgd = torch.cat([rg, d], dim=0)
            rgds.append(rgd)
        rgds = torch.stack(rgds, dim=0)
        return rgds


    @auto_fp16(apply_to=('rgb', 'depth'))
    def forward(self, rgb, img_metas, depth=None, origin_depth=None, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if return_loss:
            return self.forward_train(rgb=rgb, depth=depth, origin_depth=origin_depth, img_metas=img_metas, **kwargs)
        else:
            return self.forward_test(rgbs=rgb, depths=depth, origin_depths=origin_depth, img_metas=img_metas, **kwargs)


    def forward_train(self,
                      rgb,
                      depth,
                      origin_depth,
                      img_metas,
                      gt_rect_grasps,
                      gt_labels=None,
                      gt_depths=None,
                      gt_scores=None,
                      gt_object_ids=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_meta (list[dict]): list of image info dict where each dict has:
                'img_shape', 'scale_factor', 'flip', and my also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        gt_rect_grasps_hbb = rbbox_to_hbbox_list(gt_rect_grasps)
        gt_rect_grasps_xywhtheta = points_to_xywhtheta_list(gt_rect_grasps)

        # x = self.extract_feat(rgb)

        # rgd
        rgd = self.generate_rgd(rgb, depth)
        x = self.extract_feat(rgd) # x得到的是一个元组，对于resnet，可包含多个conv层的结果

        # x_rgb = self.rgb_backbone(rgb)
        # depth = depth.repeat(1, 3, 1, 1)
        # x_depth = self.depth_backbone(depth)
        #
        # x = []
        # for i in range(len(x_rgb)):
        #     x.append(x_rgb[i] + x_depth[i])
        # x = tuple(x)
        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_rect_grasps_hbb,
                gt_scores,
                gt_labels=None,
                gt_bboxes_ignore=None,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)

        roi_losses = self.roi_head.forward_train(x,
                                                 # origin_depth,
                                                 img_metas,
                                                 proposal_list,
                                                 gt_rect_grasps_hbb,
                                                 gt_rect_grasps_xywhtheta,
                                                 gt_depths,
                                                 gt_scores,
                                                 **kwargs)
        losses.update(roi_losses)

        if torch.isnan(losses['loss_grasp']):
            import pdb
            pdb.set_trace()

        return losses

    def forward_test(self, rgbs, img_metas, depths=None, origin_depths=None, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        for var, name in [(rgbs, 'rgb'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(rgbs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(rgbs)}) '
                             f'!= num of image meta ({len(img_metas)})')

        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        for img, img_meta in zip(rgbs, img_metas):
            batch_size = len(img_meta)
            for img_id in range(batch_size):
                img_meta[img_id]['batch_input_shape'] = tuple(img.size()[-2:])

        if num_augs == 1:
            # proposals (List[List[Tensor]]): the outer list indicates
            # test-time augs (multiscale, flip, etc.) and the inner list
            # indicates images in a batch.
            # The Tensor should have a shape Px4, where P is the number of
            # proposals.
            if 'proposals' in kwargs:
                kwargs['proposals'] = kwargs['proposals'][0]
            if depths is not None:
                depth = depths[0]
            else:
                depth = None
            if origin_depths is not None:
                origin_depth = origin_depths[0]
            else:
                origin_depth = None
            # if gt_rect_grasps is not None:
            #     gt_rect_grasp = gt_rect_grasps[0]
            #     gt_score = gt_scores[0]
            # else:
            #     gt_rect_grasp = None
            #     gt_score = None
            return self.simple_test(rgb=rgbs[0], depth=depth, origin_depth=origin_depth,
                                    img_metas=img_metas[0], **kwargs)
        else:
            assert rgbs[0].size(0) == 1, 'aug test does not support ' \
                                         'inference with batch size ' \
                                         f'{rgbs[0].size(0)}'
            # TODO: support test augmentation for predefined proposals
            assert 'proposals' not in kwargs
            return self.aug_test(rgbs, img_metas, **kwargs)



    def simple_test(self,
                    rgb,
                    depth,
                    origin_depth,
                    img_metas,
                    gt_rect_grasps=None,
                    gt_scores=None,
                    proposals=None,
                    rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."

        # dataset = 'cornell'
        dataset = 'graspnet'

        # x = self.extract_feat(rgb)
        rgd = self.generate_rgd(rgb, depth)
        x = self.extract_feat(rgd)

        proposal_list = self.rpn_head.simple_test_rpn(x,
                                                      # origin_depth,
                                                      img_metas)

        results = self.roi_head.simple_test(x,
                                            origin_depth,
                                            proposal_list, img_metas, dataset=dataset, rescale=rescale)


        if dataset == 'graspnet':
            dump_folder = self.dump_folder
            eval_root = os.path.join(os.path.abspath('.'), dump_folder)
            if not os.path.exists(eval_root):
                os.makedirs(eval_root)
            for i in range(len(results)):
                sceneId = img_metas[i]['sceneId']
                annId = img_metas[i]['annId']
                camera = img_metas[i]['camera']
                eval_dir = os.path.join(eval_root, 'scene_%04d' % sceneId, camera)
                if not os.path.exists(eval_dir):
                    os.makedirs(eval_dir)
                eval_file = os.path.join(eval_dir, "%04d.npy" %annId)
                results[i].save_npy(eval_file)

        # for ground truth
        # dump_folder = 'eval/graspnet/faster_r2cnn_r50/gt/'
        # eval_root = os.path.join(os.path.abspath('.'), dump_folder)
        # if not os.path.exists(eval_root):
        #     os.makedirs(eval_root)
        # results = []
        # gt_rect_grasps_xywhtheta = points_to_xywhtheta_list(gt_rect_grasps[0])
        # for i in range(len(img_metas)):
        #     sceneId = img_metas[i]['sceneId']
        #     annId = img_metas[i]['annId']
        #     camera = img_metas[i]['camera']
        #     eval_dir = os.path.join(eval_root, 'scene_%04d' % sceneId, camera)
        #     if not os.path.exists(eval_dir):
        #         os.makedirs(eval_dir)
        #     eval_file = os.path.join(eval_dir, "%04d.npy" %annId)
        #     if len(gt_rect_grasps_xywhtheta[i]) != len(gt_scores[0][i]):
        #         import pdb
        #         pdb.set_trace()
        #     gt_rect_grasp_xywhtheta = gt_rect_grasps_xywhtheta[i].detach().cpu().numpy()
        #     gt_score = gt_scores[0][i].detach().cpu().numpy()
        #     result_rect = xywhtheta_to_rect_grasp_group(gt_rect_grasp_xywhtheta, gt_score.reshape(-1, 1), img_metas[i]['ori_shape'])
        #     result = result_rect.to_grasp_group(camera, depth[i].squeeze().detach().cpu().numpy())
        #     results.append(result)
        #     result.save_npy(eval_file)

        return results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        proposal_list = self.aug_test_rpn(
            self.extract_feats(imgs), img_metas, self.test_cfg.rpn)
        det_bboxes, det_labels = self.aug_test_bboxes(
            self.extract_feats(imgs), img_metas, proposal_list,
            self.test_cfg.rcnn)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= img_metas[0][0]['scale_factor']
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(
                self.extract_feats(imgs), img_metas, det_bboxes, det_labels)
            return bbox_results, segm_results
        else:
            return bbox_results

