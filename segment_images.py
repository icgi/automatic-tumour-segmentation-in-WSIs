import sys
from pathlib import Path
import argparse
import time
import traceback
import logging
from collections import namedtuple
from contextlib import suppress
from typing import Any, Callable, List, Mapping, Optional

# Import numpy before pytorch in order to avoid error. The error is also present when
# numpy is not imported here. Related to https://github.com/pytorch/pytorch/issues/37377
import cv2  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.joinpath("process", "src")))

import align_tiles  # type: ignore
import configurations  # type: ignore
import data  # type: ignore
import encoders_init  # type: ignore
import merge_predictions  # type: ignore
import network  # type: ignore
import tile  # type: ignore
import utils as process_utils  # type: ignore

sys.path.insert(0, str(Path(__file__).resolve().parent.joinpath("postprocess")))

import segment_probability_maps  # type: ignore

sys.path.insert(0, str(Path(__file__).resolve().parent.joinpath("common")))

from background_filter import background_filter  # type: ignore

log = logging.getLogger("inference")

InpaintingWeights = namedtuple("InpaintingWeights", ["left", "right"])
Result = namedtuple("Result", ["input", "prediction"])


def log_progress(
    status: Optional[Mapping[str, Any]],
    has_reference: bool = False,
):
    if status is None:
        header = "{:>25}{:>12}{:>12}{:>12}".format(
            "Progress",
            "Images/sec",
            "Secs/step",
            "Time left",
        )
        if has_reference:
            header += "{:>14}{:>12}{:>19}{:>17}".format(
                "Informedness",
                "Markedness",
                "Mean Informedness",
                "Mean Markedness",
            )
        log.info(header)
    else:
        percent = status["step"] / status["num_steps"] * 100
        progress = "{:>6} / {:>6} = {:>5.1f}%".format(
            status["step"], status["num_steps"], percent
        )
        secs_left = status["sec_per_step"] * (status["num_steps"] - status["step"])
        time_left = process_utils.format_time(secs_left)
        status_str = "{:>25}{:>12,.3f}{:>12,.3f}{:>12}".format(
            progress,
            status["images_per_sec"],
            status["sec_per_step"],
            time_left,
        )
        if has_reference:
            inf = status["informedness"]
            minf = status["mean_informedness"]
            mar = status["markedness"]
            mmar = status["mean_markedness"]
            inf_str = "          None" if inf is None else f"{inf:>14,.3f}"
            mar_str = "        None" if mar is None else f"{mar:>12,.3f}"
            minf_str = "               None" if minf is None else f"{minf:>19,.3f}"
            mmar_str = "             None" if mmar is None else f"{mmar:>17,.3f}"
            status_str += f"{inf_str}{mar_str}{minf_str}{mmar_str}"
        log.info(status_str)


def write_batch(sample_batch, prediction_batch, common_dir, out_root_dir, class_labels):
    image_path_batch = sample_batch["image_path"]
    input_paths = [Path(p) for p in image_path_batch]
    out_paths = process_utils.relative_paths(input_paths, common_dir, out_root_dir)
    out_paths = [p.with_suffix(".png") for p in out_paths]
    log.debug(f"Writing probability maps to {process_utils.common_path(out_paths)}")
    crop = False
    for i, prediction in enumerate(prediction_batch):
        height = prediction.shape[-2]
        width = prediction.shape[-1]
        if height != sample_batch["height"][i] or width != sample_batch["width"]:
            crop = True
    crop_heights = sample_batch["height"] if crop else None
    crop_widths = sample_batch["width"] if crop else None
    process_utils.write_batch(
        out_paths,
        prediction_batch,
        crop_heights=crop_heights,
        crop_widths=crop_widths,
        class_labels=class_labels,
    )


def write_npy(out_path, tensor):
    np.save(
        out_path,
        tensor.detach().cpu().numpy().astype(np.float32),
        allow_pickle=False,
    )


def write_features(out_dir, net):
    out_dir.mkdir(parents=True, exist_ok=True)
    log.debug(f"Writing activations to {out_dir}")
    # NOTE: features must be manually "enabled" by storing them in a dict during
    # inference as they are normally not stored in order to save space.
    # An example of how this can be done:
    #
    # class Decoder():
    #     ...
    #     self.features = {}
    #
    #     def forward(x):
    #         self.features["x1"] = x
    #         x = operation(x)
    #         self.features["x2"] = x
    #         ...
    #
    for name, feature in net.decoder.features.items():
        write_npy(out_dir.joinpath(f"{name}.npy"), feature)


def run_network_inference(
    df: pd.DataFrame,
    conf: configurations.Configurations,
    preprocessing_fn: Callable,
    amp_autocast: Callable,
    net: Any,
    probability_map_dir: Path,
    common_input_dir: Path,
) -> List[Result]:
    dataset = data.SegmentedImages(
        df,
        conf.classes,
        conf.train_mode,
        conf.min_divisor,
        data.get_test_transform(conf),
        preprocessing_fn,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=conf.batch_size,
        shuffle=False,
        num_workers=conf.num_workers,
        drop_last=False,
    )

    start_step_time = time.time()
    log_progress(None, dataset.has_reference)
    step = 0
    steps_since_last = 0
    informedness = []
    markedness = []
    results: List[Result] = []
    for sample_batch in dataloader:
        # FIXME: Incorrect for batch_size != 1
        sample_name = Path(sample_batch["image_path"][0]).stem
        last_step = step == len(dataloader) - 1
        with torch.no_grad():
            step += 1
            steps_since_last += 1

            # NOTE: context is suppressed if not amp
            with amp_autocast():
                image_batch = sample_batch["image"].to(conf.device)
                logit_batch = (
                    net.forward(image_batch).detach().cpu().numpy().astype(np.float32)
                )
                predictions = np.floor(
                    np.clip(
                        255.0 * process_utils.logit_to_prediction_np(logit_batch),
                        0.0,
                        255.0,
                    )
                ).astype(np.uint8)
            images_per_step = predictions.shape[0]

            if conf.inference_per_scan:
                print(
                    "ERROR: run mode 'inference_per_scan' set in config is deprecated. "
                    "If you want to run inference per scan, you can use the program in "
                    "the 'segmentation/scan_segmentation' folder"
                )
                exit()
                # TODO: Remove everything with 'inference_per_scan'
                results.append(Result(sample_batch, predictions))

            if dataset.has_reference:
                prediction_masks = np.argmax(predictions, axis=1)
                reference_masks = sample_batch["mask"].cpu().numpy()
                metrics = process_utils.performance_evaluation(
                    prediction_masks,
                    reference_masks,
                )
                informedness.append(metrics["Informedness"])
                markedness.append(metrics["Markedness"])

            if not conf.inference_per_scan:
                write_batch(
                    sample_batch,
                    predictions,
                    common_input_dir,
                    probability_map_dir,
                    class_labels=conf.classes,
                )

            if conf.write_features:
                feature_dir = conf.debug_dir.joinpath("features", sample_name)
                feature_dir.mkdir(parents=True, exist_ok=True)
                write_npy(feature_dir.joinpath("input.npy"), image_batch)
                write_features(feature_dir, net)

            sec_per_step = time.time() - start_step_time

            if step % conf.monitor_progress == 0 or step <= 10 or last_step:
                sec_per_step = (time.time() - start_step_time) / steps_since_last
                status = {
                    "step": step,
                    "num_steps": len(dataloader),
                    "images_per_sec": images_per_step / sec_per_step,
                    "sec_per_step": sec_per_step,
                }
                if dataset.has_reference:
                    nonnan_inf = [x for x in informedness if x is not None]
                    nonnan_mar = [x for x in markedness if x is not None]
                    mean_inf = None if len(nonnan_inf) == 0 else np.mean(nonnan_inf)
                    mean_mar = None if len(nonnan_mar) == 0 else np.mean(nonnan_mar)
                    status["informedness"] = informedness[-1]
                    status["mean_informedness"] = mean_inf
                    status["markedness"] = markedness[-1]
                    status["mean_markedness"] = mean_mar
                log_progress(status, dataset.has_reference)
                steps_since_last = 0
                start_step_time = time.time()

    return results


def get_foreground_mask(parent_path: Path) -> Optional[np.ndarray]:
    """
    Return foreground mask image if it exists.
    Return full image shape, either from foreground mask or image
    """
    mask_candidates = list(parent_path.glob("*mask*foreground*png"))
    if len(mask_candidates) > 0:
        foreground_mask = cv2.imread(str(mask_candidates[0]), cv2.IMREAD_GRAYSCALE) > 0
        height, width = foreground_mask.shape
    else:
        image_path = parent_path.joinpath(parent_path.name + ".png")
        if image_path.exists():
            full_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            foreground_mask = background_filter(image_bgr=full_image) > 0
        else:
            log.warning(f"Trying to read non-existing image {image_path}")
            foreground_mask = None
    return foreground_mask


def class_index_to_label(mask, class_labels):
    labels = np.unique(mask)
    assert np.max(labels) < len(class_labels), "Incompatible label and classes"
    for label in sorted(labels):
        mask[mask == label] = class_labels[label]
    return mask


def run_per_scan_segmentation(
    df: pd.DataFrame,
    conf: configurations.Configurations,
    preprocessing_fn: Callable,
    amp_autocast: Callable,
    net: Any,
    probability_map_dir: Path,
    common_input_dir: Path,
    full_image_root_dir: Optional[Path],
):
    # TODO: This is a mess. Refactor into functions.
    total_tile_count = len(df)
    existing_tile_count = np.sum(df["Exists"].values)
    not_existing_tile_count = total_tile_count - existing_tile_count
    log.debug(f"Total number of tiles:        {total_tile_count:>3}")
    log.debug(f"Number of tiles existing:     {existing_tile_count:>3}")
    log.debug(f"Number of tiles not existing: {not_existing_tile_count:>3}")
    image_folder_path = Path(df["ImageFolder"].values[0])
    scan_name = image_folder_path.name
    if conf.merge_tile_predictions:
        merged_output_root = conf.probability_map_dir.parent.joinpath(
            "merged_probability_maps"
        )
        merged_output_path = merged_output_root.joinpath(
            image_folder_path.relative_to(common_input_dir)
        ).with_name(scan_name + ".png")

    existing_tiles = {}
    if not conf.overwrite and conf.merge_tile_predictions:
        # Get resize factor for tiles
        # Get foreground mask
        ranges = [tile.Range2D(p.stem) for p in df["ImagePath"].values]
        shape_from_tiles = tile.global_dim_from_ranges(ranges)
        if full_image_root_dir is not None:
            full_image_dir = full_image_root_dir.joinpath(
                image_folder_path.relative_to(common_input_dir)
            ).parent
            foreground_mask = get_foreground_mask(full_image_dir)
        else:
            foreground_mask = None

        if foreground_mask is not None:
            full_height, full_width = foreground_mask.shape
            resize_factor = max(
                full_height / shape_from_tiles[0],
                full_width / shape_from_tiles[1],
            )
        else:
            full_height = shape_from_tiles[0]
            full_width = shape_from_tiles[1]
            foreground_mask = np.ones((full_height, full_width), dtype=np.uint8)
            resize_factor = None

        existing = df[df["Exists"]]["Output"].values
        df = df[~df["Exists"]]
        if (not (len(df) == 0 and merged_output_path.exists())) and len(existing) > 0:
            # These are only needed if we are merging. But if all already exists
            # (len(df) == 0), and not overwrite, we are not computing a new merged
            # probability maps if it already exists, and therefore we would like to skip
            # this scan fast. Also, if no tile predictions exist, we cannot construct
            # tiles from them.
            log.debug(f"Reading {len(existing)} existing tile predictions")
            existing_tiles = tile.construct_tiles(
                existing,
                [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in existing],
                resize_factor,
            )
        else:
            log.debug("Don't read any existing tile predictions")

    if len(df) > 0:
        log.debug("Run network inference")
        results = run_network_inference(
            df,
            conf,
            preprocessing_fn,
            amp_autocast,
            net,
            conf.probability_map_dir,
            common_input_dir,
        )

        if conf.write_tile_predictions:
            for result in results:
                write_batch(
                    result.input,
                    result.prediction,
                    common_input_dir,
                    conf.probability_map_dir,
                    class_labels=conf.classes,
                )
    else:
        log.debug("Don't compute new tile predictions")
        results = []

    if conf.merge_tile_predictions:
        # NOTE: Assumes that the class of interest is in the last channel
        if len(results) > 0:
            input_paths = [Path(r.input["image_path"][0]) for r in results]
            tiles = tile.construct_tiles(
                input_paths,
                [r.prediction[0, -1, :, :] for r in results],
                resize_factor,
            )
        else:
            tiles = {}
        if not conf.overwrite and merged_output_path.exists() and len(tiles) == 0:
            log.debug("Skip merging")
        else:
            log.debug("Merge tile predictions")
            tiles.update(existing_tiles)
            if conf.align_tiles:
                # TODO: Construct background mask per tile(?)
                tiles = align_tiles.align_tiles(
                    tiles,
                    foreground_mask,
                    85.0,
                    85.0,
                )
            merged_image = merge_predictions.merge_tile_predictions(tiles)
            if full_height is not None and full_width is not None:
                merged_image = cv2.resize(
                    merged_image,
                    (full_width, full_height),
                    interpolation=cv2.INTER_AREA,
                )
            log.debug(f"Writing merged probability map to {merged_output_path}")
            merged_output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(merged_output_path), merged_image)

            if conf.segment_merged_prediction:
                segmented_output_root = conf.probability_map_dir.parent.joinpath(
                    "segmentation_masks", conf.segmentation_name
                )
                segmented_output_path = segmented_output_root.joinpath(
                    image_folder_path.relative_to(common_input_dir)
                ).with_name(scan_name + ".png")
                if conf.overwrite or (
                    not conf.overwrite and not segmented_output_path.exists()
                ):
                    segmented_output_path.parent.mkdir(parents=True, exist_ok=True)
                    log.debug("Segment probability map")
                    mask = segment_probability_maps.segment_image(
                        merged_image,
                        None,
                        foreground_mask,
                        conf.postprocess_smooth,
                        conf.postprocess_method,
                        conf.postprocess_prune,
                        None,
                        conf.postprocess_close_holes,
                    )
                    mask = class_index_to_label(mask, [0, 255])
                    log.debug(f"Writing segmentation mask to {segmented_output_path}")
                    cv2.imwrite(str(segmented_output_path), mask)


def segment_images(conf: configurations.Configurations):
    preprocessing_fn = encoders_init.get_preprocessing_fn(
        conf.encoder,
        conf.initialise_encoder,
        conf.input_space,
        conf.input_range,
        conf.train_mean,
        conf.train_std,
    )
    df = data.get_data_paths(
        conf.input_data_path,
        conf.probability_map_dir,
        conf.overwrite,
        conf.merge_tile_predictions,
    )
    if len(df) == 0:
        log.info("No data to process. Terminating")
        sys.exit()

    dataset_size = len(df)
    images_per_step = conf.batch_size
    number_of_steps = dataset_size / images_per_step
    dataset_name = conf.input_data_path.stem
    image_folders = sorted(list(set(df["ImageFolder"].values)))
    image_folder_count = len(image_folders)
    log.info(f"Input {dataset_name}")
    log.info(f"Number of gpus:     {conf.num_gpus:>7}")
    log.info(f"Dataset size:       {dataset_size:>7}")
    log.info(f"Image folders:      {image_folder_count:>7}")
    log.info(f"Images per step:    {images_per_step:>7}")
    log.info(f"Number of steps:    {number_of_steps:>10,.2f}")

    amp_autocast = torch.cuda.amp.autocast if conf.amp else suppress

    net = network.load(conf)
    process_utils.network_summary(
        conf.log_dir,
        net,
        conf.logger,
        conf.encoder,
        conf.distributed,
        [conf.batch_size, 3, conf.target_height, conf.target_width],
    )

    log.info(f"Start segmenting {dataset_name}")
    common_input_dir = process_utils.common_path(df["ImagePath"].values)
    if len(df["ImagePath"].values[0].relative_to(common_input_dir).parts) == 1:
        # If df only has tiles from one case, this condition is true, and we force the
        # output tile paths to have a case folder as parent
        common_input_dir = common_input_dir.parent
    if conf.inference_per_scan:
        datasets = {
            f"{'/'.join(f.parts[-4:-1])}": df[df["ImageFolder"] == f]
            for f in image_folders
        }
        for i, (name, scan_df) in enumerate(datasets.items()):
            log.info(f"{i + 1} / {image_folder_count}: {name}")
            run_per_scan_segmentation(
                scan_df,
                conf,
                preprocessing_fn,
                amp_autocast,
                net,
                conf.probability_map_dir,
                common_input_dir,
                conf.full_image_path,
            )
    else:
        _ = run_network_inference(
            df,
            conf,
            preprocessing_fn,
            amp_autocast,
            net,
            conf.probability_map_dir,
            common_input_dir,
        )


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(
        description="",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    conf = configurations.set_up(parser, False)
    log.info("Configurations are set up, start program")

    try:
        segment_images(conf)
    except Exception as err:
        log.error(f"Program finished with exception: \n{err}")
        log.error(traceback.format_exc())
    else:
        log.info("Program finished correctly")
    finally:
        formatted = process_utils.format_time(time.time() - start_time)
        log.info(f"Elapsed time: {formatted}")


if __name__ == "__main__":
    main()
