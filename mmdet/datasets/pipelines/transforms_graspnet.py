from ..builder import PIPELINES
import mmcv
import numpy as np
import numpy.random as random
import cv2
import copy

@PIPELINES.register_module()
class ResizeGraspNet(object):
    """Resize images & bbox & mask.

    This transform resizes the input image to some scale. Bboxes and masks are
    then resized with the same scale factor. If the input dict contains the key
    "scale", then the scale in the input dict is used, otherwise the specified
    scale in the init method is used. If the input dict contains the key
    "scale_factor" (if MultiScaleFlipAug does not give img_scale but
    scale_factor), the actual scale will be computed by image shape and
    scale_factor.

    `img_scale` can either be a tuple (single-scale) or a list of tuple
    (multi-scale). There are 3 multiscale modes:

    - ``ratio_range is not None``: randomly sample a ratio from the ratio \
      range and multiply it with the image scale.
    - ``ratio_range is None`` and ``multiscale_mode == "range"``: randomly \
      sample a scale from the multiscale range.
    - ``ratio_range is None`` and ``multiscale_mode == "value"``: randomly \
      sample a scale from multiple scales.

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
        multiscale_mode (str): Either "range" or "value".
        ratio_range (tuple[float]): (min_ratio, max_ratio)
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image.
        bbox_clip_border (bool, optional): Whether clip the objects outside
            the border of the image. Defaults to True.
        backend (str): Image resize backend, choices are 'cv2' and 'pillow'.
            These two backends generates slightly different results. Defaults
            to 'cv2'.
        override (bool, optional): Whether to override `scale` and
            `scale_factor` so as to call resize twice. Default False. If True,
            after the first resizing, the existed `scale` and `scale_factor`
            will be ignored so the second resizing can be allowed.
            This option is a work-around for multiple times of resize in DETR.
            Defaults to False.
    """

    def __init__(self,
                 with_rgb=True,
                 with_depth=True,
                 with_origin_depth=False,
                 with_segment=False,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True,
                 backend='cv2',
                 override=False,
                 EPS = 1e-8):
        self.with_rgb = with_rgb
        self.with_depth = with_depth
        self.with_origin_depth = with_origin_depth
        self.with_segment = with_segment
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert mmcv.is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            # mode 1: given a scale and a range of image ratio
            assert len(self.img_scale) == 1
        else:
            # mode 2: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.backend = backend
        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        # TODO: refactor the override option in Resize
        self.override = override
        self.EPS = EPS

    @staticmethod
    def random_select(img_scales):
        """Randomly select an img_scale from given candidates.

        Args:
            img_scales (list[tuple]): Images scales for selection.

        Returns:
            (tuple, int): Returns a tuple ``(img_scale, scale_dix)``, \
                where ``img_scale`` is the selected image scale and \
                ``scale_idx`` is the selected index in the given candidates.
        """

        assert mmcv.is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        """Randomly sample an img_scale when ``multiscale_mode=='range'``.

        Args:
            img_scales (list[tuple]): Images scale range for sampling.
                There must be two tuples in img_scales, which specify the lower
                and uper bound of image scales.

        Returns:
            (tuple, None): Returns a tuple ``(img_scale, None)``, where \
                ``img_scale`` is sampled scale and None is just a placeholder \
                to be consistent with :func:`random_select`.
        """

        assert mmcv.is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(
            min(img_scale_long),
            max(img_scale_long) + 1)
        short_edge = np.random.randint(
            min(img_scale_short),
            max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        """Randomly sample an img_scale when ``ratio_range`` is specified.

        A ratio will be randomly sampled from the range specified by
        ``ratio_range``. Then it would be multiplied with ``img_scale`` to
        generate sampled scale.

        Args:
            img_scale (tuple): Images scale base to multiply with ratio.
            ratio_range (tuple[float]): The minimum and maximum ratio to scale
                the ``img_scale``.

        Returns:
            (tuple, None): Returns a tuple ``(scale, None)``, where \
                ``scale`` is sampled ratio multiplied with ``img_scale`` and \
                None is just a placeholder to be consistent with \
                :func:`random_select`.
        """

        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        """Randomly sample an img_scale according to ``ratio_range`` and
        ``multiscale_mode``.

        If ``ratio_range`` is specified, a ratio will be sampled and be
        multiplied with ``img_scale``.
        If multiple scales are specified by ``img_scale``, a scale will be
        sampled according to ``multiscale_mode``.
        Otherwise, single scale will be used.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: Two new keys 'scale` and 'scale_idx` are added into \
                ``results``, which would be used by subsequent pipelines.
        """

        if self.ratio_range is not None:
            scale, scale_idx = self.random_sample_ratio(
                self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results['scale'] = scale
        results['scale_idx'] = scale_idx

    def _resize_rgb_depth(self, results):
        """Resize images with ``results['scale']``."""
        if self.keep_ratio:
            if self.with_rgb:
                rgb, scale_factor = mmcv.imrescale(
                    results['rgb'],
                    results['scale'],
                    return_scale=True,
                    backend=self.backend)
            if self.with_depth:
                depth, _ = mmcv.imrescale(
                    results['depth'],
                    results['scale'],
                    return_scale=True,
                    interpolation='nearest',
                    backend=self.backend)
            if self.with_origin_depth:
                origin_depth, _ = mmcv.imrescale(
                    results['origin_depth'],
                    results['scale'],
                    return_scale=True,
                    interpolation='nearest',
                    backend=self.backend)
            if self.with_segment:
                segment, _ = mmcv.imrescale(
                    results['segment'],
                    results['scale'],
                    return_scale=True,
                    interpolation='nearest',
                    backend=self.backend)
            # the w_scale and h_scale has minor difference
            # a real fix should be done in the mmcv.imrescale in the future
            new_h, new_w = rgb.shape[:2]
            h, w = results['rgb'].shape[:2]
            w_scale = new_w / w
            h_scale = new_h / h
        else:
            if self.with_rgb:
                rgb, w_scale, h_scale = mmcv.imresize(
                    results['rgb'],
                    results['scale'],
                    return_scale=True,
                    backend=self.backend)
            if self.with_depth:
                depth, _, _ = mmcv.imresize(
                    results['depth'],
                    results['scale'],
                    return_scale=True,
                    interpolation='nearest',
                backend=self.backend)
            if self.with_origin_depth:
                origin_depth, _ = mmcv.imrescale(
                    results['origin_depth'],
                    results['scale'],
                    return_scale=True,
                    interpolation='nearest',
                    backend=self.backend)
            if self.with_segment:
                segment, _ = mmcv.imrescale(
                    results['segment'],
                    results['scale'],
                    return_scale=True,
                    interpolation='nearest',
                    backend=self.backend)

        if self.with_rgb:
            results['rgb'] = rgb
        if self.with_depth:
            results['depth'] = depth
        if self.with_origin_depth:
            results['origin_depth'] = origin_depth
        if self.with_segment:
            results['segment'] = segment

        scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                dtype=np.float32)
        results['img_shape'] = rgb.shape
        # in case that there is no padding
        results['pad_shape'] = rgb.shape
        results['scale_factor'] = scale_factor
        results['keep_ratio'] = self.keep_ratio

    def _resize_grasps(self, results):
        if len(results['rect_grasp_fields']) == 0:
            return
        scale_factor = results['scale_factor']
        scale_factor = np.tile(
            (scale_factor[0], scale_factor[1]), 4).astype(results['gt_rect_grasps'].dtype)
        grasps = results['gt_rect_grasps'] * scale_factor
        results['gt_rect_grasps'] = grasps


    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor', \
                'keep_ratio' keys are added into result dict.
        """

        if 'scale' not in results:
            if 'scale_factor' in results:
                img_shape = results['img'].shape[:2]
                scale_factor = results['scale_factor']
                assert isinstance(scale_factor, float)
                results['scale'] = tuple(
                    [int(x * scale_factor) for x in img_shape][::-1])
            else:
                self._random_scale(results)
        else:
            if not self.override:
                assert 'scale_factor' not in results, (
                    'scale and scale_factor cannot be both set.')
            else:
                results.pop('scale')
                if 'scale_factor' in results:
                    results.pop('scale_factor')
                self._random_scale(results)

        self._resize_rgb_depth(results)
        self._resize_grasps(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'multiscale_mode={self.multiscale_mode}, '
        repr_str += f'ratio_range={self.ratio_range}, '
        repr_str += f'keep_ratio={self.keep_ratio}, '
        repr_str += f'grasp_clip_border={self.grasp_clip_border})'
        return repr_str

@PIPELINES.register_module()
class RandomFlipGraspNet(object):
    """Flip the image & bbox & mask.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    When random flip is enabled, ``flip_ratio``/``direction`` can either be a
    float/string or tuple of float/string. There are 3 flip modes:

    - ``flip_ratio`` is float, ``direction`` is string: the image will be
        ``direction``ly flipped with probability of ``flip_ratio`` .
        E.g., ``flip_ratio=0.5``, ``direction='horizontal'``,
        then image will be horizontally flipped with probability of 0.5.
    - ``flip_ratio`` is float, ``direction`` is list of string: the image wil
        be ``direction[i]``ly flipped with probability of
        ``flip_ratio/len(direction)``.
        E.g., ``flip_ratio=0.5``, ``direction=['horizontal', 'vertical']``,
        then image will be horizontally flipped with probability of 0.25,
        vertically with probability of 0.25.
    - ``flip_ratio`` is list of float, ``direction`` is list of string:
        given ``len(flip_ratio) == len(direction)``, the image wil
        be ``direction[i]``ly flipped with probability of ``flip_ratio[i]``.
        E.g., ``flip_ratio=[0.3, 0.5]``, ``direction=['horizontal',
        'vertical']``, then image will be horizontally flipped with probability
         of 0.3, vertically with probability of 0.5

    Args:
        flip_ratio (float | list[float], optional): The flipping probability.
            Default: None.
        direction(str | list[str], optional): The flipping direction. Options
            are 'horizontal', 'vertical', 'diagonal'. Default: 'horizontal'.
            If input is a list, the length must equal ``flip_ratio``. Each
            element in ``flip_ratio`` indicates the flip probability of
            corresponding direction.
    """

    def __init__(self, with_rgb=True,
                 with_depth=True,
                 with_origin_depth=False,
                 with_segment=False,
                 flip_ratio=None,
                 direction='horizontal'):
        self.with_rgb = with_rgb
        self.with_depth = with_depth
        self.with_origin_depth = with_origin_depth
        self.with_segment = with_segment
        if isinstance(flip_ratio, list):
            assert mmcv.is_list_of(flip_ratio, float)
            assert 0 <= sum(flip_ratio) <= 1
        elif isinstance(flip_ratio, float):
            assert 0 <= flip_ratio <= 1
        elif flip_ratio is None:
            pass
        else:
            raise ValueError('flip_ratios must be None, float, '
                             'or list of float')
        self.flip_ratio = flip_ratio

        valid_directions = ['horizontal', 'vertical', 'diagonal']
        if isinstance(direction, str):
            assert direction in valid_directions
        elif isinstance(direction, list):
            assert mmcv.is_list_of(direction, str)
            assert set(direction).issubset(set(valid_directions))
        else:
            raise ValueError('direction must be either str or list of str')
        self.direction = direction

        if isinstance(flip_ratio, list):
            assert len(self.flip_ratio) == len(self.direction)

    def grasp_flip(self, grasps, img_shape, direction):
        """Flip bboxes horizontally.

        Args:
            bboxes (numpy.ndarray): Bounding boxes, shape (..., 4*k)
            img_shape (tuple[int]): Image shape (height, width)
            direction (str): Flip direction. Options are 'horizontal',
                'vertical'.

        Returns:
            numpy.ndarray: Flipped bounding boxes.
        """

        assert grasps.shape[-1] % 8 == 0
        flipped = grasps.copy()
        if direction == 'horizontal':
            w = img_shape[1]
            flipped[..., 0] = w - grasps[..., 2]
            flipped[..., 1] = grasps[..., 3]
            flipped[..., 2] = w - grasps[..., 0]
            flipped[..., 3] = grasps[..., 1]
            flipped[..., 4] = w - grasps[..., 6]
            flipped[..., 5] = grasps[..., 7]
            flipped[..., 6] = w - grasps[..., 4]
            flipped[..., 7] = grasps[..., 5]
        else:
            raise ValueError(f"Invalid flipping direction '{direction}'")
        return flipped

    def __call__(self, results):
        """Call function to flip bounding boxes, masks, semantic segmentation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added \
                into result dict.
        """

        if 'flip' not in results:
            if isinstance(self.direction, list):
                # None means non-flip
                direction_list = self.direction + [None]
            else:
                # None means non-flip
                direction_list = [self.direction, None]

            if isinstance(self.flip_ratio, list):
                non_flip_ratio = 1 - sum(self.flip_ratio)
                flip_ratio_list = self.flip_ratio + [non_flip_ratio]
            else:
                non_flip_ratio = 1 - self.flip_ratio
                # exclude non-flip
                single_ratio = self.flip_ratio / (len(direction_list) - 1)
                flip_ratio_list = [single_ratio] * (len(direction_list) -
                                                    1) + [non_flip_ratio]

            cur_dir = np.random.choice(direction_list, p=flip_ratio_list)

            results['flip'] = cur_dir is not None
        if 'flip_direction' not in results:
            results['flip_direction'] = cur_dir
        if results['flip']:
            # flip rgb
            if self.with_rgb:
                results['rgb'] = mmcv.imflip(
                    results['rgb'], direction=results['flip_direction'])
            # flip depth
            if self.with_depth:
                results['depth'] = mmcv.imflip(results['depth'], direction=results['flip_direction'])
            if self.with_origin_depth:
                results['origin_depth'] = mmcv.imflip(results['origin_depth'], direction=results['flip_direction'])
            if self.with_segment:
                results['segment'] = mmcv.imflip(results['segment'], direction=results['flip_direction'])
            # flip grasps
            results['gt_rect_grasps'] = self.grasp_flip(results['gt_rect_grasps'],
                                               results['img_shape'],
                                               results['flip_direction'])
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(flip_ratio={self.flip_ratio})'


@PIPELINES.register_module()
class NormalizeRGB(object):
    """Normalize the image.

    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        results['rgb'] = mmcv.imnormalize(results['rgb'], self.mean, self.std,
                                            self.to_rgb)
        results['rgb_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str

@PIPELINES.register_module()
class NormalizeDepth(object):
    """Normalize the image.

    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        # img = results['depth'].copy().astype(np.float32)
        # assert img.dtype != np.uint8
        # mean = np.float64(self.mean.reshape(1, -1))
        # stdinv = 1 / np.float64(self.std.reshape(1, -1))
        # cv2.subtract(img, mean, img)  # inplace
        # cv2.multiply(img, stdinv, img)  # inplace
        # return img
        results['depth'] = mmcv.imnormalize(results['depth'], self.mean, self.std,
                                            to_rgb=False)
        results['depth_norm_cfg'] = dict(
            mean=self.mean, std=self.std)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std})'
        return repr_str


@PIPELINES.register_module()
class PadGraspNet(object):
    """Pad the image & mask.

    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",

    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, with_rgb=True, with_depth=True, with_origin_depth=False,
                 with_segment=False, size=None, size_divisor=None,
                 rgb_pad_val=0, depth_pad_val=0, segment_pad_val=0):
        self.with_rgb = with_rgb
        self.with_depth = with_depth
        self.with_origin_depth = with_origin_depth
        self.with_segment = with_segment
        self.size = size
        self.size_divisor = size_divisor
        self.rgb_pad_val = rgb_pad_val
        self.depth_pad_val = depth_pad_val
        self.segment_pad_val = segment_pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_rgb_depth(self, results):
        """Pad images according to ``self.size``."""
        if self.size is not None:
            if self.with_rgb:
                padded_rgb = mmcv.impad(
                    results['rgb'], shape=self.size, pad_val=self.rgb_pad_val)
            if self.with_depth:
                padded_depth = mmcv.impad(
                    results['depth'], shape=self.size, pad_val=self.depth_pad_val)
            if self.with_origin_depth:
                padded_origin_depth = mmcv.impad(
                    results['origin_depth'], shape=self.size, pad_val=self.depth_pad_val)
            if self.with_segment:
                padded_segment = mmcv.impad(
                    results['segment'], shape=self.size, pad_val=self.segment_pad_val)
        elif self.size_divisor is not None:
            if self.with_rgb:
                padded_rgb = mmcv.impad_to_multiple(
                    results['rgb'], self.size_divisor, pad_val=self.rgb_pad_val)
            if self.with_depth:
                padded_depth = mmcv.impad_to_multiple(
                    results['depth'], self.size_divisor, pad_val=self.depth_pad_val)
            if self.with_origin_depth:
                padded_origin_depth = mmcv.impad_to_multiple(
                    results['origin_depth'], self.size_divisor, pad_val=self.depth_pad_val)
            if self.with_segment:
                padded_segment = mmcv.impad_to_multiple(
                    results['segment'], self.size_divisor, pad_val=self.segment_pad_val)

        if self.with_rgb:
            results['rgb'] = padded_rgb
        if self.with_depth:
            results['depth'] = padded_depth
        if self.with_origin_depth:
            results['origin_depth'] = padded_origin_depth
        if self.with_segment:
            results['segment'] = padded_segment
        results['pad_shape'] = padded_rgb.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """
        self._pad_rgb_depth(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str

@PIPELINES.register_module()
class RandomCropKeepBoxGraspNet(object):
    def __init__(self,
                 with_rgb=True,
                 with_depth=True):
        self.with_rgb = with_rgb
        self.with_depth = with_depth

    def __call__(self, results):
        if self.with_rgb:
            rgb = results['rgb']
        if self.with_depth:
            depth = results['depth']
        rect_grasps = results['gt_rect_grasps']
        height, width, _ = rgb.shape
        # Get the minimum boundary of all bboxes in dim x and y
        xmin = np.min(rect_grasps[:, 0::2])
        ymin = np.min(rect_grasps[:, 1::2])
        # Get the maximum boundary of all bboxes in dim x and y
        xmax = np.max(rect_grasps[:, 0::2])
        ymax = np.max(rect_grasps[:, 1::2])

        # Get top left corner's coordinate of the crop box
        x_start = int(random.uniform(0, xmin))
        y_start = int(random.uniform(0, ymin))
        # Get lower right corner corner's coordinate of the crop box
        x_end = int(random.uniform(xmax, width))
        y_end = int(random.uniform(ymax, height))

        # Crop the image
        rgb = rgb[y_start:y_end, x_start:x_end]
        img_shape = rgb.shape
        results['img_shape'] = img_shape
        if self.with_rgb:
            results['rgb'] = rgb
        if self.with_depth:
            results['depth'] = depth

        # Adjust the grasps to fit the cropped image
        rect_grasps[:, 0::2] -= x_start
        rect_grasps[:, 1::2] -= y_start
        results['gt_rect_grasps'] = rect_grasps

        return results

@PIPELINES.register_module()
class CenterCropGraspNet(object):
    def __init__(self,
                 size=(351, 351),
                 with_rgb=True,
                 with_depth=True):
        self.size = size
        self.with_rgb = with_rgb
        self.with_depth = with_depth

    def __call__(self, results):
        if self.with_rgb:
            rgb = results['rgb']
        if self.with_depth:
            depth = results['depth']

        height, width, _ = rgb.shape
        x_center = int(width / 2)
        y_center = int(height / 2)
        w_new = self.size[0]
        h_new = self.size[1]

        # Get top left corner's coordinate of the crop box
        x_start = int(x_center - w_new / 2)
        y_start = int(y_center - h_new / 2)
        # Get lower right corner corner's coordinate of the crop box
        x_end = int(x_center + w_new / 2)
        y_end = int(y_center + h_new / 2)

        # Crop the image
        rgb = rgb[y_start:y_end, x_start:x_end]
        img_shape = rgb.shape
        results['img_shape'] = img_shape
        if self.with_rgb:
            results['rgb'] = rgb
        if self.with_depth:
            results['depth'] = depth

        # Adjust the grasps to fit the cropped image
        if 'gt_rect_grasps' in results:
            rect_grasps = results['gt_rect_grasps']
            rect_grasps[:, 0::2] -= x_start
            rect_grasps[:, 1::2] -= y_start
            results['gt_rect_grasps'] = rect_grasps

        results['center_crop'] = True
        results['center_crop_xstart'] = x_start
        results['center_crop_ystart'] = y_start

        return results

@PIPELINES.register_module()
class RandomRotateGraspNet(object):
    def __init__(self,
                 angle,
                 rotate_ratio,
                 with_rgb=True,
                 with_depth=True,
                 with_origin_depth=False,
                 with_segment=False,
                 debug=False,
                 modify=False):
        self.angle = angle
        self.with_rgb = with_rgb
        self.with_depth = with_depth
        self.with_origin_depth = with_origin_depth
        self.with_segment = with_segment
        self.debug = debug
        self.rotate_ratio = rotate_ratio
        self.modify = modify

    def rotate_poly(self, h, w, new_h ,new_w, poly, rotate_matrix_T):
        poly_temp = np.zeros(poly.shape)
        poly_temp[0::2] = poly[0::2] - (w - 1) * 0.5
        poly_temp[1::2] = poly[1::2] - (h - 1) * 0.5
        coords = poly_temp.reshape(4, 2)
        new_coords = np.matmul(coords, rotate_matrix_T) + np.array([(new_w - 1) * 0.5, (new_h - 1) * 0.5])
        rotated_polys = new_coords.reshape(-1, )

        return rotated_polys

    def __call__(self, results):
        if random.random() < self.rotate_ratio:
            return results
        # if random.random() < 0.5:
        #     angle_base = self.angle[0]
        # else:
        #     angle_base = self.angle[0] + 180
        if self.with_rgb:
            rgb = results['rgb']
        if self.with_depth:
            depth = results['depth']
        if self.with_origin_depth:
            origin_depth = results['origin_depth']
        if self.with_segment:
            segment = results['segment']
        a = random.random() * (self.angle[1] - self.angle[0]) + self.angle[0] # 在给定角度范围内生成一个随机旋转角
        s = 1 #缩放比
        h, w = rgb.shape[:2]
        center = ((w - 1) * 0.5, (h - 1) * 0.5)
        M = cv2.getRotationMatrix2D(angle=a, center=center, scale=s)
        M_T = copy.deepcopy(M[:2, :2]).T
        if self.with_rgb:
            rgb = cv2.warpAffine(rgb, M, (w, h))
            results['rgb'] = rgb
            if self.debug:
                cv2.imwrite(r'/home/qinran_2020/debug.png', rgb)
        if self.with_depth:
            depth = cv2.warpAffine(depth, M, (w, h))
            results['depth'] = depth
        if self.with_origin_depth:
            origin_depth = cv2.warpAffine(origin_depth, M, (w, h))
            results['origin_depth'] = origin_depth
        if self.with_segment:
            segment = cv2.warpAffine(segment, M, (w, h))
            results['segment'] = segment
        results['img_shape'] = rgb.shape

        rect_grasps = results['gt_rect_grasps']
        rotate_rect_grasps = []
        for i in range(len(rect_grasps)):
            rotate_rect_grasp = self.rotate_poly(h, w, h, w, rect_grasps[i], M_T)
            rotate_rect_grasps.append(rotate_rect_grasp)
        rotate_rect_grasps = np.array(rotate_rect_grasps).astype(np.float32)
        if self.debug:
            pts = []
            for i in range(50):
                pt = rotate_rect_grasps[i].reshape((-1, 1, 2)).astype(np.int32)
                pts.append(pt)
            cv2.polylines(rgb, pts, True, (0, 255, 255))
            cv2.imwrite(r'/home/qinran_2020/debug.png', rgb)

        if self.modify:
            # 去掉图像外的抓取
            scores = results['gt_scores']
            depths = results['gt_depths']
            object_ids = results['gt_object_ids']
            center_x = (rotate_rect_grasps[:, 0] + rotate_rect_grasps[:, 2] + rotate_rect_grasps[:, 4] + rotate_rect_grasps[:, 6]) / 4
            center_y = (rotate_rect_grasps[:, 1] + rotate_rect_grasps[:, 3] + rotate_rect_grasps[:, 5] + rotate_rect_grasps[:, 7]) / 4
            valid_mask1 = np.logical_and(center_x < w, center_x >= 0)
            valid_mask2 = np.logical_and(center_y < h,  center_y >= 0)
            valid_mask = np.logical_and(valid_mask1, valid_mask2)

            results['gt_rect_grasps'] = rotate_rect_grasps[valid_mask]
            results['gt_depths'] = depths[valid_mask]
            results['gt_scores'] = scores[valid_mask]
            results['gt_object_ids'] = object_ids[valid_mask]
        else:
            results['gt_rect_grasps'] = rotate_rect_grasps

        return results

