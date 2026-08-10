"""Blockwise mask generation for the optional CLS-token reconstruction loss.

`MaskingGenerator` follows the DINOv2 / iBOT block-masking implementation (Apache 2.0).

`MaskSampleTransform` is the wrapper that runs in the dataloader workers
and stamps one block-mask per timestep onto the sample dict under `mask`.
"""

import math
import random

import numpy as np
import torch


class MaskingGenerator:
    """Generate block masks on a (height, width) patch grid.

    Iteratively places random-aspect-ratio rectangles until the target
    number of patches is masked. Adapted from
    https://github.com/facebookresearch/dinov2 (Apache 2.0).
    """

    def __init__(
        self,
        input_size,
        num_masking_patches=None,
        min_num_patches=4,
        max_num_patches=None,
        min_aspect=0.3,
        max_aspect=None,
    ):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches

        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def get_shape(self):
        return self.height, self.width

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for _ in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)

                num_masked = mask[top : top + h, left : left + w].sum()
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                if delta > 0:
                    break
        return delta

    def __call__(self, num_masking_patches=0):
        mask = np.zeros(shape=self.get_shape(), dtype=bool)
        mask_count = 0
        while mask_count < num_masking_patches:
            max_mask_patches = num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)

            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta

        return mask


class MaskSampleTransform:
    """Per-sample transform that attaches blockwise patch masks to a sample dict.

    For each timestep in `sample["pixels"]` of shape (T, C, H, W), samples a
    masking ratio uniformly from [min_ratio, max_ratio] and produces a
    (gh, gw) bool mask. The stacked (T, gh, gw) tensor is written to
    `sample["mask"]` for downstream consumption by the CLS-recon loss.
    """

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        min_ratio: float = 0.1,
        max_ratio: float = 0.5,
        source: str = "pixels",
        target: str = "mask",
    ):
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        grid = img_size // patch_size
        self.grid = grid
        self.n_tokens = grid * grid
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.source = source
        self.target = target
        # max_num_patches caps the size of any single rectangle the generator
        # places; set to max_ratio * n_tokens.
        self.generator = MaskingGenerator(
            input_size=grid,
            max_num_patches=int(max_ratio * self.n_tokens),
        )

    def __call__(self, sample: dict) -> dict:
        pixels = sample[self.source]
        T = pixels.shape[0]
        masks = np.zeros((T, self.grid, self.grid), dtype=bool)
        for t in range(T):
            ratio = random.uniform(self.min_ratio, self.max_ratio)
            num_mask = int(self.n_tokens * ratio)
            masks[t] = self.generator(num_mask)
        sample[self.target] = torch.from_numpy(masks)
        return sample
