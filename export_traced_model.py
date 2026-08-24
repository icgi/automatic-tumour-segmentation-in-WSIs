"""
Trace a trained segmentation model with `torch.jit` and save it for inference from
libtorch.

The traced module takes a uint8 RGB tile (N, H, W, 3) and returns uint8 probability maps
(N, C, H, W), so normalisation and the conversion of logits to 8-bit probabilities are
part of the graph. See `process/src/traced_model.py`.

Tracing happens inside the same autocast context that `segment_images.py` uses, which
records the casts autocast inserted as ordinary operations in the graph. The saved graph
is therefore an explicit mixed-precision graph and must be run *outside* autocast.

Two things are required for the saved graph to reproduce the python results exactly, and
both are checked by this program:

- The autocast weight cache must be off while tracing. A cached cast is not recorded,
  and the tracer either stores a stale constant or fails.
- Graph executor optimisation must be off when running. Fusing operations changes the
  order of arithmetic, and in half precision that changes the result. In c++ this is
  `torch::jit::setGraphExecutorOptimize(false)`.

Example:

    python export_traced_model.py --config process/config/inference.toml \\
        --restore /path/to/model.tar --output /path/to/model.pt
"""

import argparse
import contextlib
import logging
import sys
from collections import namedtuple
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import cv2  # type: ignore
import numpy as np
import toml  # type: ignore
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.joinpath("process", "src")))

import configurations  # type: ignore
import encoders_init  # type: ignore
import network  # type: ignore
import traced_model  # type: ignore
import utils as process_utils  # type: ignore

log = logging.getLogger("export")

# A probability at or below this is background, where a difference cannot show
BACKGROUND = 0.002

# The quantised output, the same without quantisation, and the eager counterparts
Models = namedtuple(
    "Models", ["traced", "traced_probability", "wrapper", "probability", "network"]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-c",
        "--config",
        metavar="PATH",
        type=Path,
        required=True,
        help="Inference config file (.toml). Must be the one used for inference",
    )
    parser.add_argument(
        "-r",
        "--restore",
        metavar="PATH",
        type=Path,
        required=True,
        help="Filepath to the checkpoint holding the model to trace",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        type=Path,
        required=True,
        help="Filepath of the traced model (.pt)",
    )
    parser.add_argument(
        "-g",
        "--gpu",
        metavar="INT",
        type=int,
        default=0,
        help="Which gpu to trace on [0]",
    )
    parser.add_argument(
        "-s",
        "--size",
        metavar="INT",
        type=int,
        nargs=2,
        default=[1024, 1024],
        help="Height and width to trace with [1024 1024]. The graph accepts other\n"
        "sizes, which is verified on a second size",
    )
    parser.add_argument(
        "-t",
        "--tile",
        metavar="PATH",
        type=Path,
        help="Image to verify on, centre cropped to --size. Random pixels by default,\n"
        "which drive the network towards a near constant output and make the\n"
        "reported fractions almost meaningless",
    )
    parser.add_argument(
        "--class_index",
        metavar="INT",
        type=int,
        help="Return the probability map of this class only, indexing `classes`\n"
        "in the config. Use -1 for the class the merging step reads.\n"
        "All classes by default",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        metavar="INT",
        type=int,
        default=3,
        help="Verbosity level [3]",
    )
    return parser.parse_args()


def set_up_config(args: argparse.Namespace) -> configurations.Configurations:
    conf = configurations.Configurations()
    with args.config.open("r") as ifile:
        conf.update(toml.loads(ifile.read()))
    conf.train_mode = False
    conf.restore_path = args.restore
    conf.num_gpus = 1
    conf.device = f"cuda:{args.gpu}"
    assert torch.cuda.is_available(), "CUDA is unavailable"
    assert conf.input_space == "RGB", f"Unsupported input space {conf.input_space}"
    assert list(conf.input_range) == [0, 1], f"Unsupported range {conf.input_range}"
    torch.cuda.set_device(args.gpu)
    return conf


def autocast(conf: configurations.Configurations) -> Any:
    """A fresh context per use. Entering one twice reuses its cached weight casts."""
    if not conf.amp:
        return contextlib.nullcontext()
    return torch.cuda.amp.autocast(cache_enabled=False)


def reference_logits(
    net: Any,
    conf: configurations.Configurations,
    preprocessing_fn: Callable,
    tile: np.ndarray,
) -> np.ndarray:
    """The network output as `segment_images.py` obtains it, for the same input tile."""
    image = preprocessing_fn(tile).transpose(2, 0, 1).astype("float32")
    # The dataloader collates into a contiguous batch, and convolution results depend on
    # the memory format
    image = np.ascontiguousarray(image[np.newaxis])
    with torch.no_grad(), autocast(conf):
        batch = torch.from_numpy(image).to(conf.device)
        return net(batch).detach().float().cpu().numpy()


def reference_prediction(
    net: Any,
    conf: configurations.Configurations,
    preprocessing_fn: Callable,
    tile: np.ndarray,
) -> np.ndarray:
    """The prediction as `segment_images.py` computes it, for the same input tile."""
    logits = reference_logits(net, conf, preprocessing_fn, tile)
    probability = process_utils.logit_to_prediction_np(logits)
    return np.floor(np.clip(255.0 * probability, 0.0, 255.0)).astype(np.uint8)


def make_tile(
    image: Optional[np.ndarray], size: Tuple[int, int], seed: int
) -> np.ndarray:
    """A tile of `size`, centre cropped from `image`, or random pixels without one."""
    if image is None:
        rng = np.random.default_rng(seed)
        return rng.integers(0, 256, size=(size[0], size[1], 3), dtype=np.uint8)
    assert image.shape[0] >= size[0], f"Image height below {size[0]}"
    assert image.shape[1] >= size[1], f"Image width below {size[1]}"
    top = (image.shape[0] - size[0]) // 2
    left = (image.shape[1] - size[1]) // 2
    return np.ascontiguousarray(image[top:top + size[0], left:left + size[1]])


def compare(name: str, got: np.ndarray, want: np.ndarray) -> int:
    differing = int(np.count_nonzero(got != want))
    max_diff = int(
        np.abs(got.astype(np.int16) - want.astype(np.int16)).max() if got.size else 0
    )
    percent = 100.0 * differing / got.size
    log.info(
        f"{name:<44}{differing:>12,} / {got.size:<14,} "
        f"= {percent:>8.5f}%  max {max_diff} level(s)"
    )
    return differing


def verify(
    models: "Models",
    conf: configurations.Configurations,
    preprocessing_fn: Callable,
    tile: np.ndarray,
) -> bool:
    """
    Check the saved graph against the wrapped network it was traced from, and against
    the python inference pipeline. Repeated runs come first, since a difference between
    two runs of the same code puts an upper bound on what any comparison can show.
    """
    traced, wrapper, net = models.traced, models.wrapper, models.network
    log.info(f"Verifying at {tile.shape[0]} x {tile.shape[1]}")
    batch = torch.from_numpy(tile).unsqueeze(0).to(conf.device)

    # cuDNN takes the first convolution algorithm whose workspace fits in the memory
    # free at the time of the call. Each path is run once before being compared, to
    # compare them in the state that a run over many tiles settles into.
    with torch.no_grad(), autocast(conf):
        _ = wrapper(batch)
    with torch.no_grad(), torch.jit.optimized_execution(False):
        _ = traced(batch)

    first = reference_logits(net, conf, preprocessing_fn, tile)
    second = reference_logits(net, conf, preprocessing_fn, tile)
    log.info(
        f"{'Network logits over two runs':<44}"
        f"largest difference {np.abs(first - second).max():.3e}"
    )

    with torch.no_grad(), autocast(conf):
        expected = wrapper(batch).cpu().numpy()
        repeated = wrapper(batch).cpu().numpy()
    with torch.no_grad(), torch.jit.optimized_execution(False):
        prediction = traced(batch).cpu().numpy()
        prediction_repeated = traced(batch).cpu().numpy()

    _ = compare("Wrapped network over two runs", repeated, expected)
    _ = compare("Traced over two runs", prediction_repeated, prediction)
    _ = compare("Traced vs wrapped network", prediction, expected)

    with torch.no_grad(), autocast(conf):
        eager_probability = models.probability(batch)
    with torch.no_grad(), torch.jit.optimized_execution(False):
        traced_probability = models.traced_probability(batch)
    gap = (eager_probability - traced_probability).abs().max().item()
    covered = 100.0 * (eager_probability > BACKGROUND).float().mean().item()
    log.info(f"{'Probability gap traced vs wrapped':<44}{gap:.3e}")
    log.info(f"{'Tile above ' + str(BACKGROUND):<44}{covered:.1f}%")
    if covered < 1.0:
        log.warning(
            "Almost all of this tile is background, where the probability saturates "
            "and hides any difference. Pass --tile with tissue on it"
        )
    # One step of the half precision logits moves a probability by about 2e-3, so a gap
    # below one 8-bit level is the last bit of the network rather than a different graph
    ok = gap < 1.0 / 255.0
    if not ok:
        log.error("The saved graph does not reproduce the network it was traced from")

    # The graph executor only fuses operations from the third call onwards
    with torch.no_grad():
        for _ in range(3):
            optimised = traced(batch).cpu().numpy()
    fused = compare("Traced with fusion enabled vs disabled", optimised, prediction)
    if fused != 0:
        log.warning(
            "Graph executor optimisation changes the result. Disable it when running "
            "the model, in c++ with torch::jit::setGraphExecutorOptimize(false)"
        )

    reference = reference_prediction(net, conf, preprocessing_fn, tile)
    if wrapper.class_index is not None:
        reference = np.take(reference, [wrapper.class_index], axis=1)
    _ = compare("Traced vs python inference pipeline", prediction, reference)
    return ok


def export(args: argparse.Namespace) -> bool:
    conf = set_up_config(args)
    settings = encoders_init.get_preprocessing_params(
        conf.encoder,
        conf.initialise_encoder,
        conf.input_space,
        conf.input_range,
        conf.train_mean,
        conf.train_std,
    )
    preprocessing_fn = encoders_init.get_preprocessing_fn(
        conf.encoder,
        conf.initialise_encoder,
        conf.input_space,
        conf.input_range,
        conf.train_mean,
        conf.train_std,
    )
    if args.class_index is not None:
        assert -len(conf.classes) <= args.class_index < len(conf.classes), (
            f"--class_index {args.class_index} is outside classes {conf.classes}"
        )
    log.info(f"Encoder            {conf.encoder}")
    log.info(f"Decoder            {conf.decoder}")
    output_classes = (
        conf.classes if args.class_index is None else [conf.classes[args.class_index]]
    )
    log.info(f"Classes            {conf.classes}")
    log.info(f"Output classes     {output_classes}")
    log.info(f"Mixed precision    {conf.amp}")
    log.info(f"Mean               {settings['mean']}")
    log.info(f"Std                {settings['std']}")

    net = network.load(conf)
    # A second copy, so that the reference predictions come from an unmodified network
    traceable = network.load(conf)
    replaced = traced_model.replace_dynamic_padding(traceable)
    log.info(f"Replaced {replaced} pooling layer(s) with fixed padding")
    wrapper = traced_model.TileSegmentation(
        traceable, settings["mean"], settings["std"], args.class_index
    ).eval()
    # Parameters that require grad are cast through the autocast cache, which hides the
    # cast from the tracer
    for parameter in wrapper.parameters():
        parameter.requires_grad_(False)

    height, width = args.size
    image = None
    if args.tile is not None:
        bgr = cv2.imread(str(args.tile), cv2.IMREAD_COLOR)
        assert bgr is not None, f"Could not read {args.tile}"
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    first_tile = make_tile(image, (height, width), height * width)
    example = torch.zeros((1, height, width, 3), dtype=torch.uint8, device=conf.device)
    log.info(f"Tracing at {height} x {width}")
    with torch.no_grad(), autocast(conf):
        traced = torch.jit.trace(wrapper, example, check_trace=False)

    reduced_values = sum(
        1
        for node in traced.inlined_graph.nodes()
        for output in node.outputs()
        if output.type().kind() == "TensorType"
        and str(output.type().dtype()) in ["torch.float16", "torch.bfloat16"]
    )
    log.info(f"Reduced precision values in graph: {reduced_values}")
    if conf.amp and reduced_values == 0:
        log.warning(
            "Found no reduced precision in the graph, which means autocast was not "
            "captured. The comparison against the wrapped network below decides"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(traced, str(args.output))
    log.info(f"Wrote traced model to {args.output}")

    probability = traced_model.TileProbability(wrapper).eval()
    with torch.no_grad(), autocast(conf):
        traced_probability = torch.jit.trace(probability, example, check_trace=False)

    loaded = torch.jit.load(str(args.output), map_location=conf.device)
    models = Models(loaded, traced_probability, wrapper, probability, net)
    ok = verify(models, conf, preprocessing_fn, first_tile)
    # A scan has tiles of several sizes, so the graph must not be tied to the traced one
    second_size = (height - conf.min_divisor, width - 2 * conf.min_divisor)
    second_tile = make_tile(image, second_size, second_size[0] * second_size[1])
    ok = verify(models, conf, preprocessing_fn, second_tile) and ok
    return ok


def main():
    args = parse_args()
    configurations.setup_logger(args.verbose)
    if export(args):
        log.info("Program finished correctly")
    else:
        log.error("Program finished with errors")
        sys.exit(1)


if __name__ == "__main__":
    main()
