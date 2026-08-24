"""
Wrapping of a segmentation network for tracing with `torch.jit`, so that inference can
be run from libtorch.

The wrapper includes everything the python inference pipeline does around the network,
that is input normalisation and conversion of logits to 8-bit probabilities. A traced
graph is therefore the complete tile-to-probability-map computation, and no part of it
has to be reimplemented in c++.

Run this file to check that the wrapper reproduces the numpy implementation of those
surrounding steps.
"""

from typing import Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F


def replace_dynamic_padding(module: nn.Module) -> int:
    """
    Replace timm's `AvgPool2dSame` with an equivalent module padding by a fixed amount.

    `AvgPool2dSame` computes its padding from the input size. Part of that expression
    goes through `math.ceil`, which the tracer records as a constant while the rest
    stays tied to the input, so the traced graph pads wrongly for any other input size.
    A scan has tiles of several sizes, so this has to be resolved before tracing.

    For a 2 x 2 kernel with stride 1 the padding is one column and one row regardless of
    input size. Only that case occurs here, and it is asserted.
    """
    from timm.models.layers import AvgPool2dSame  # type: ignore

    targets = [
        (parent, name)
        for parent in module.modules()
        for name, child in parent.named_children()
        if isinstance(child, AvgPool2dSame)
    ]
    for parent, name in targets:
        pool = getattr(parent, name)
        assert tuple(pool.kernel_size) == (2, 2), f"Kernel size {pool.kernel_size}"
        assert tuple(pool.stride) == (1, 1), f"Stride {pool.stride}"
        setattr(
            parent,
            name,
            nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.AvgPool2d(
                    pool.kernel_size,
                    pool.stride,
                    ceil_mode=pool.ceil_mode,
                    count_include_pad=pool.count_include_pad,
                ),
            ),
        )
    return len(targets)


class TileSegmentation(nn.Module):
    """
    Input:  uint8 tensor (N, H, W, 3), RGB channel order, as read from disk.
    Output: uint8 tensor (N, C, H, W), one probability map per class, or (N, 1, H, W)
            for the class at `class_index`.

    Valid for `input_space = "RGB"` and `input_range = [0, 1]` only.
    """

    def __init__(
        self,
        net: nn.Module,
        mean: Sequence[float],
        std: Sequence[float],
        class_index: Optional[int] = None,
    ):
        super().__init__()
        self.net = net
        self.class_index = class_index
        # float64 reproduces the numpy preprocessing in data.SegmentedImages bit for bit
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float64))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float64))

    def normalise(self, image: torch.Tensor) -> torch.Tensor:
        # Made contiguous while still uint8 to keep the network input in the same memory
        # format as a tensor from the dataloader. Convolution results depend on it.
        x = image.permute(0, 3, 1, 2).contiguous().to(torch.float32)
        # One channel at a time, so the float64 copy is a third of the tile. uint8
        # values are exact in both float types, so this is the same as converting all.
        for channel in range(3):
            values = x[:, channel].to(torch.float64)
            values.div_(255.0).sub_(self.mean[channel]).div_(self.std[channel])
            x[:, channel].copy_(values)
        return x

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        logits = self.net(self.normalise(image))
        # Softmax normalises over the classes, so all of them are computed even when
        # one probability map is returned. Selecting before exp keeps log_softmax
        # itself, and therefore its result, unchanged.
        log_probability = F.log_softmax(logits.to(torch.float32), dim=1)
        if self.class_index is not None:
            log_probability = log_probability[:, self.class_index].unsqueeze(1)
        # In place from here, log_softmax being the last operation needing a tensor
        probability = log_probability.exp_()
        return probability.mul_(255.0).clamp_(0.0, 255.0).floor_().to(torch.uint8)


def self_check():
    """Compare wrapper input and output against the numpy implementations."""
    import numpy as np

    import data
    import encoders_init
    import utils

    mean = [8.297992e-01, 7.106879e-01, 8.241846e-01]
    std = [1.051075e-01, 1.543867e-01, 9.917571e-02]
    preprocessing_fn = encoders_init.get_preprocessing_fn(
        "timm-eca-nfnet-l3", None, "RGB", [0, 1], mean, std
    )
    rng = np.random.default_rng(0)
    tile = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)

    # The network is the identity, so the wrapper is exactly the steps around it
    wrapper = TileSegmentation(nn.Identity(), mean, std).eval()
    with torch.no_grad():
        batch = torch.from_numpy(tile).unsqueeze(0)
        traced = torch.jit.trace(wrapper, batch, check_trace=False)
        normalised = wrapper.normalise(batch).numpy()
        prediction = traced(batch).numpy()

    reference_input = data.transpose_to_float32(preprocessing_fn(tile))
    assert np.array_equal(normalised[0], reference_input), "Normalisation differs"

    reference_output = np.floor(
        np.clip(
            255.0 * utils.logit_to_prediction_np(reference_input[np.newaxis]),
            0.0,
            255.0,
        )
    ).astype(np.uint8)
    differing = np.count_nonzero(prediction != reference_output)
    max_level_diff = np.abs(
        prediction.astype(np.int16) - reference_output.astype(np.int16)
    ).max()
    # log_softmax and exp are not bit-exact across implementations, so the 8-bit
    # probability of a pixel can land one level off
    assert max_level_diff <= 1, f"Probabilities differ by {max_level_diff} levels"
    assert differing < 1e-4 * prediction.size, f"{differing} probabilities differ"
    print(f"OK: {differing} / {prediction.size} probabilities one level off")


if __name__ == "__main__":
    self_check()
