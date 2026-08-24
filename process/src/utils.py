from collections import defaultdict, OrderedDict
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import albumentations as albu
import cv2
import torchinfo
import scipy.special

log = logging.getLogger("utils")


def format_time(total_seconds: int) -> str:
    total_hours = total_seconds / 60 / 60
    total_minutes = total_seconds / 60
    hours = int(np.floor(total_hours))
    minutes = int(np.floor(total_minutes - hours * 60))
    seconds = int(np.round(total_seconds - minutes * 60 - hours * 60 * 60))
    if hours > 0:
        formatted = f"{hours}h {minutes:2}m {seconds:2}s"
    elif minutes > 0:
        formatted = f"{minutes:2}m {seconds:2}s"
    elif seconds > 0:
        formatted = f"{seconds:2}s"
    else:
        formatted = f"{total_seconds:.2f}s"
    return formatted


def update_progress(status: Mapping[str, Any], out_path: Path):
    header = "Step,Epoch,Loss,"
    if "loss_2" in status.keys():
        header += "Loss1,Loss2,"
    header += "ImagesPerSec,SecPerStep,StepLength"
    content_list = [status["step"], status["epoch"], status["loss"]]
    if "loss_2" in status.keys():
        content_list.extend([status["loss_1"], status["loss_2"]])
    content_list.extend(
        [
            status["images_per_sec"],
            status["sec_per_step"],
            status["step_length"],
        ]
    )
    content = ",".join([str(c) for c in content_list])
    if not out_path.exists():
        with out_path.open("w") as ofile:
            ofile.write(f"{header}\n")
            ofile.write(f"{content}\n")
    else:
        with out_path.open("a") as ofile:
            ofile.write(f"{content}\n")


def split_directory_with_time(directory: Path) -> Tuple[str, str]:
    """
    Assumes the input directory is on the form

    /path/to/<name>_<time>

    and that <time> is on the format

    <year>-<month>-<day>_<hour>-<minute>-<second>

    all dates are in decimal numbers, and are expected to have the length (they are
    zero-padded)

    4-2-2_2-2-2

    everything is created after year 2000, so it is likely that the first number in the
    time part will be 2.

    return <name>, <time>
    """

    basename = directory.name
    if len(basename.split("_")) < 3:
        raise ValueError("Found no <date>_<time> in directory name")
    if basename.split("_")[-2][0] != "2":
        raise ValueError("Date not in the third milennium AD.")
    name = "_".join(basename.split("_")[:-2])
    datetime = "_".join(basename.split("_")[-2:])
    return name, datetime


def make_rundir(parent: Path, date_time: str) -> Path:
    """Appends run_x parent and returns the result.

    The x in run_x will be one greater than the greatest x found in parent. If the
    folder given by parent is empty, this function will initialise the run with
    run_00, and return parent/run_00_<date_time>.

    if <parent> is empty:
        return <parent>/run_00_<date_time>
    if content of <parent> is
        <parent>
            |- run_00_some-date-time-0
            |- run_01_some-date-time-1
            |_ run_02_some-date-time-2
        return <parent>/run_03_<date_time>
    """
    version = -1
    for run_dir in parent.iterdir():
        new_version = int(run_dir.name.split("_")[1])
        if new_version > version:
            version = new_version
    result_dir = parent.joinpath(f"run_{version+1:02}_{date_time}")
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir


def common_path(paths: Sequence[Path]) -> Path:
    common_parent = paths[0].parent
    if len(set(paths)) > 1:
        for path in paths[1:]:
            common_parent = common(common_parent, path)
    return common_parent


def common(path_1: Path, path_2: Path) -> Path:
    path = Path()
    for (component_1, component_2) in zip(path_1.parts, path_2.parts):
        if component_1 == component_2:
            if path is None:
                path = Path(component_1)
            else:
                path = path.joinpath(component_1)
        else:
            break
    return path


def relative_path(old_path: Path, old_root: Path, new_root: Path) -> Path:
    return new_root.joinpath(old_path.relative_to(old_root))


def relative_paths(
    old_paths: Sequence[Path], old_root: Path, new_root: Path
) -> List[Path]:
    return [relative_path(p, old_root, new_root) for p in old_paths]


def metrics_from_counts(
    tp: int, fn: int, fp: int, tn: int, ignore_missing: bool = True
) -> Dict[str, Optional[float]]:
    n = tp + fn + tn + fp
    rp = tp + fn
    rn = tn + fp
    pp = tp + fp
    pn = tn + fn

    default = None if ignore_missing else 0

    rrp = rp / n if n > 0 else default
    rrn = rn / n if n > 0 else default
    rpp = pp / n if n > 0 else default
    rpn = pn / n if n > 0 else default
    tpr = tp / rp if rp > 0 else default
    tnr = tn / rn if rn > 0 else default
    ppv = tp / pp if pp > 0 else default
    npv = tn / pn if pn > 0 else default

    jac = None
    if tp + fn + fp > 0:
        jac = tp / (tp + fn + fp)
    else:
        jac = default

    mcc = None
    prod = np.sqrt(rp) * np.sqrt(rn) * np.sqrt(pp) * np.sqrt(pn)
    if prod > 0:
        mcc = (tp * tn - fp * fn) / prod
    else:
        mcc = default

    inf = None
    if tpr is not None and tnr is not None:
        inf = tpr + tnr - 1
    else:
        inf = default

    mar = None
    if ppv is not None and npv is not None:
        mar = ppv + npv - 1
    else:
        mar = default

    return {
        "rrp": rrp,
        "rrn": rrn,
        "rpp": rpp,
        "rpn": rpn,
        "tpr": tpr,
        "tnr": tnr,
        "ppv": ppv,
        "npv": npv,
        "jac": jac,
        "mcc": mcc,
        "inf": inf,
        "mar": mar,
    }


def performance_evaluation(
    prediction_masks: np.ndarray,
    reference_masks: np.ndarray,
    sample_average: bool = True,
    ignore_missing: bool = True,
) -> Dict[str, Any]:
    """Compute some evaluation metrics.

    Input
    prediction_masks: Batch of predicted labels for each class.
        torch tensor: [N, H, W] with values in {0, 1}
    reference_masks: Batch of reference segmentations.
        torch tensor: [N, H, W] with values in {0, 1}
    sample_average:
        True -> Compute metrics on each sample and average
        False -> Count tp, fp, fn, tn on all images and then compute derived metrics
    """
    # TODO: Filter out references with only background
    # NOTE: Only measuring foreground class
    predicted_labels = set(np.unique(prediction_masks))
    assert predicted_labels.difference(set([0, 1])) == set()
    assert len(prediction_masks.shape) == 3
    reference_labels = set(np.unique(reference_masks))
    assert reference_labels.difference(set([0, 1])) == set()
    assert len(reference_masks.shape) == 3

    batch_size = reference_masks.shape[0]

    counts: Dict[str, List] = defaultdict(list)
    for i in range(batch_size):
        fg_ref = reference_masks[i, :, :] == 1
        fg_pred = prediction_masks[i, :, :] == 1

        tp = (fg_pred & fg_ref).sum()
        tn = ((~fg_pred) & (~fg_ref)).sum()
        fp = fg_pred.sum() - tp
        fn = (~fg_pred).sum() - tn

        counts["tp"].append((fg_pred & fg_ref).sum())
        counts["tn"].append(((~fg_pred) & (~fg_ref)).sum())
        counts["fp"].append(fg_pred.sum() - tp)
        counts["fn"].append((~fg_pred).sum() - tn)

    if sample_average:
        rrp_l = []
        rpp_l = []
        tpr_l = []
        tnr_l = []
        ppv_l = []
        npv_l = []
        jac_l = []
        mcc_l = []
        inf_l = []
        mar_l = []
        for i in range(batch_size):
            # Only include metrics that are not None
            tp = counts["tp"][i]
            tn = counts["tn"][i]
            fp = counts["fp"][i]
            fn = counts["fn"][i]

            metrics = metrics_from_counts(tp, fn, fp, tn, ignore_missing)

            if metrics["rrp"] is not None:
                rrp_l.append(metrics["rrp"])
            if metrics["rpp"] is not None:
                rpp_l.append(metrics["rpp"])
            if metrics["tpr"] is not None:
                tpr_l.append(metrics["tpr"])
            if metrics["tnr"] is not None:
                tnr_l.append(metrics["tnr"])
            if metrics["ppv"] is not None:
                ppv_l.append(metrics["ppv"])
            if metrics["npv"] is not None:
                npv_l.append(metrics["npv"])
            if metrics["jac"] is not None:
                jac_l.append(metrics["jac"])
            if metrics["mcc"] is not None:
                mcc_l.append(metrics["mcc"])
            if metrics["inf"] is not None:
                inf_l.append(metrics["inf"])
            if metrics["mar"] is not None:
                mar_l.append(metrics["mar"])

        rrp = None if len(rrp_l) == 0 else np.mean(rrp_l)
        rpp = None if len(rpp_l) == 0 else np.mean(rpp_l)
        tpr = None if len(tpr_l) == 0 else np.mean(tpr_l)
        tnr = None if len(tnr_l) == 0 else np.mean(tnr_l)
        ppv = None if len(ppv_l) == 0 else np.mean(ppv_l)
        npv = None if len(npv_l) == 0 else np.mean(npv_l)
        jac = None if len(jac_l) == 0 else np.mean(jac_l)
        mcc = None if len(mcc_l) == 0 else np.mean(mcc_l)
        inf = None if len(inf_l) == 0 else np.mean(inf_l)
        mar = None if len(mar_l) == 0 else np.mean(mar_l)
    else:
        tp = sum(counts["tp"])
        tn = sum(counts["tn"])
        fp = sum(counts["fp"])
        fn = sum(counts["fn"])
        metrics = metrics_from_counts(tp, fn, fp, tn, ignore_missing)
        rrp = metrics["rrp"]
        rpp = metrics["rpp"]
        tpr = metrics["tpr"]
        tnr = metrics["tnr"]
        ppv = metrics["ppv"]
        npv = metrics["npv"]
        jac = metrics["jac"]
        mcc = metrics["mcc"]
        inf = metrics["inf"]
        mar = metrics["mar"]

    return {
        "Relative reference positive": rrp,
        "Relative predicted positive": rpp,
        "Sensitivity": tpr,
        "Specificity": tnr,
        "Positive predictive value": ppv,
        "Negative predictive value": npv,
        "Jaccard index": jac,
        "Matthews correlation": mcc,
        "Informedness": inf,
        "Markedness": mar,
    }


def class_mappings(classes: Sequence[int]) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    The input is a list with class labels. The position in the list is the class index,
    and the value at that position class label. This function creates convenience dicst
    to map between label and index.
    """
    index_to_label = {}
    label_to_index = {}
    for index, label in enumerate(classes):
        index_to_label[index] = label
        label_to_index[label] = index
    return index_to_label, label_to_index


def add_class_names(classes: Sequence[int]) -> Dict[int, str]:
    assert set(classes).difference({0, 127, 255}) == set()
    mapping = {0: "Background", 127: "Tissue without tumour", 255: "Tissue with tumour"}
    return {c: mapping[c] for c in classes}


def scale(
    array: np.ndarray,
    new_min: int = 0,
    new_max: int = 255,
    old_min: Optional[int] = None,
    old_max: Optional[int] = None,
) -> np.ndarray:
    if old_min is None:
        old_min = np.min(array)
    if old_max is None:
        old_max = np.max(array)
    return ((new_max - array) * new_min + (array - new_min) * new_max) / (
        old_max - old_min
    )


def draw_contour(
    image: np.ndarray,
    mask: np.ndarray,
    color: Sequence[int],
    thickness: int = 5,
    connectivity: int = 4,
) -> np.ndarray:
    """Draw contours of the mask on the image

    This is guaranteed to delineate the interior of the mask contour, also if line
    thickness is > 1

    image is assumed to be of type uint8
    mask is assumed to be binary with 0 for background and 255 for foreground.
    """
    if connectivity == 4:
        selem_type = cv2.MORPH_RECT
    else:
        selem_type = cv2.MORPH_CROSS
    selem = cv2.getStructuringElement(selem_type, (3, 3))
    # mask = mask > 0
    delineation = np.zeros_like(mask)
    for _ in range(thickness):
        erosion = cv2.erode(mask, selem)
        delineation = np.logical_or(mask - erosion, delineation)
        mask = erosion

    if len(image.shape) == 2:
        image = cv2.merge((image, image, image))
    image[delineation > 0, :] = np.array(color)

    return image


def overlay_masks_hsv(
    image_rgb: np.ndarray, reference_mask: np.ndarray, prediction_mask: np.ndarray
) -> np.ndarray:
    """
    Partition image into four regions:

        Reference Prediction Result
        No        No         grayscale original
        No        Yes        red overlaying grayscale original
        Yes       No         blue overlaying grayscale original
        Yes       Yes        green overlaying grayscale original
    """
    reference_mask = reference_mask == 255
    prediction_mask = prediction_mask == 255
    image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    hue, saturation, value = [image_hsv[:, :, i] for i in range(3)]
    hue[~reference_mask & ~prediction_mask] = 0
    hue[~reference_mask & prediction_mask] = 180
    hue[reference_mask & ~prediction_mask] = 100
    hue[reference_mask & prediction_mask] = 40
    image_hsv = np.dstack((hue, saturation, value))
    image_rgb = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)
    image_rgb = draw_contour(
        image_rgb, (reference_mask * 255).astype(np.uint8), [0, 0, 0], 2
    )
    image_rgb = draw_contour(
        image_rgb, (prediction_mask * 255).astype(np.uint8), [0, 0, 0], 2
    )
    image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)
    return image_rgb


def overlay_masks_rgb(
    image_rgb: np.ndarray, reference_mask: np.ndarray, prediction_mask: np.ndarray
) -> np.ndarray:
    """
    Partition image into four regions:

        Reference Prediction Result
        No        No         grayscale original
        No        Yes        red overlaying grayscale original
        Yes       No         blue overlaying grayscale original
        Yes       Yes        green overlaying grayscale original
    """
    weight = 0.8

    # Create grayscale image
    red, green, blue = cv2.split(image_rgb)
    image_gl = red / 3 + green / 3 + blue / 3
    image_gl = cv2.merge((image_gl, image_gl, image_gl))
    image_rgb = image_gl

    # Color the various masked regions
    red, green, blue = cv2.split(image_rgb)
    merged_red = red
    merged_green = green
    merged_blue = blue
    reference_mask = reference_mask == 255
    prediction_mask = prediction_mask == 255

    mask = reference_mask & prediction_mask
    merged_green[mask] = (1 - weight) * green[mask] + weight * 255
    mask = reference_mask & ~prediction_mask
    merged_blue[mask] = (1 - weight) * blue[mask] + weight * 255
    mask = ~reference_mask & prediction_mask
    merged_red[mask] = (1 - weight) * red[mask] + weight * 255

    image_rgb = cv2.merge((merged_red, merged_green, merged_blue))
    image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)

    return image_rgb


def write_batch(
    out_paths: Sequence[Path],
    predictions: np.ndarray,
    images: Optional[np.ndarray] = None,
    references: Optional[np.ndarray] = None,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
    crop_heights: Optional[Sequence[int]] = None,
    crop_widths: Optional[Sequence[int]] = None,
    class_labels: Sequence[int] = [0, 255],
):
    """
    out_paths: base filepaths corresponding to the output. [N]
    predictions: normalised network outputs. [N, C, H, W] with u8 values
    images: input images after preprocessing. [N, 3, H, W]
    references: reference masks. [N, C, H, W]
    """
    crop = crop_heights is not None and crop_widths is not None

    assert predictions.dtype == np.uint8, "Predictions assumed to be uint8"
    if images is not None:
        # Useful for debugging input batches and resulting predictions
        assert len(images.shape) == 4, "Expected shape (N, C, H, W)"
        assert len(out_paths) == len(images), "Should be one path per image"
        for i, image in enumerate(images):
            if crop:
                assert crop_heights is not None
                assert crop_widths is not None
                crop_height: Optional[int] = crop_heights[i]
                crop_width: Optional[int] = crop_widths[i]
            else:
                crop_height = None
                crop_width = None
            out_path = out_paths[i]
            filename_prefix = f"{i:02d}_{out_path.stem}"
            out_dir = out_path.parent
            out_dir.mkdir(parents=True, exist_ok=True)

            image = image.transpose((1, 2, 0))
            if mean is not None and std is not None:
                image = np.clip(255 * ((image * std) + mean), 0, 255).astype(np.uint8)
            if crop:
                image = albu.augmentations.crops.functional.center_crop(
                    image, crop_height, crop_width
                )
            filename = filename_prefix + "_input.png"
            cv2.imwrite(
                str(out_dir.joinpath(filename)), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            )

            for class_label in class_labels:
                class_index = class_labels.index(class_label)

                if references is not None:
                    if len(references.shape) == 3:
                        reference_mask = (references[i, :, :] == class_index).astype(
                            int
                        )
                    else:
                        reference_mask = references[i, class_index, :, :]
                    reference_mask = np.floor(255 * reference_mask).astype(np.uint8)
                    if crop:
                        reference_mask = (
                            albu.augmentations.crops.functional.center_crop(
                                reference_mask, crop_height, crop_width
                            )
                        )
                    filename = (
                        filename_prefix + f"_label-{class_label}_reference-mask.png"
                    )
                    cv2.imwrite(str(out_dir.joinpath(filename)), reference_mask)

                prediction = predictions[i, class_index, :, :]
                prediction_mask = 255 * (prediction >= 128).astype(np.uint8)
                if crop:
                    prediction_mask = albu.augmentations.crops.functional.center_crop(
                        prediction_mask, crop_height, crop_width
                    )

                filename = filename_prefix + f"_label-{class_label}_prediction.png"
                cv2.imwrite(str(out_dir.joinpath(filename)), prediction)

                image = overlay_masks_rgb(image, reference_mask, prediction_mask)
                filename = filename_prefix + f"_label-{class_label}_result.png"
                cv2.imwrite(
                    str(out_dir.joinpath(filename)),
                    cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                )
    else:
        # Used to write predictions during inference
        folder_name = "probability_maps_class-255"
        assert len(out_paths) == len(predictions), "Should be one path per prediction"
        for class_label in class_labels:
            class_index = class_labels.index(class_label)
            for i, prediction in enumerate(predictions):
                class_prediction = prediction[class_index, :, :]
                if crop:
                    assert crop_heights is not None
                    assert crop_widths is not None
                    crop_height = crop_heights[i]
                    crop_width = crop_widths[i]
                    class_prediction = albu.augmentations.crops.functional.center_crop(
                        class_prediction, crop_height, crop_width
                    )
                temp_out_path = out_paths[i]
                new_folder_name = f"probability_maps_class-{class_label}"
                parts = list(temp_out_path.parts)
                assert folder_name in parts
                parts[parts.index(folder_name)] = new_folder_name
                out_path = Path(*parts)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_path), class_prediction)


def maybe_remove_module_prefix(state_dict: Mapping[str, Any]) -> Dict[str, Any]:
    """
    If a nn.DataParallel model has been saved, checkpoint keys are prefixed with
    "module.".

    In inference, one can either wrap the network in nn.DataParallel, or remove
    "module." from the key list. The latter is done by this function. See e.g. issue at

    https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/3
    """
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]
        new_state_dict[key] = value
    return new_state_dict


def from_timm_universal(state_dict: Mapping[str, Any]) -> Dict[str, Any]:
    """
    When moving between timm universal models and custom model, this can be used.

    Example:

    Model without timm universal    | Checkpoint from timm universal
    encoder.model.stem_conv1.weight | encoder.stem.conv1.weight
    """

    def split_underscore_element(key_parts, prefix):
        candidates = [p for p in key_parts if p.startswith(prefix)]
        assert len(candidates) == 1
        element = candidates[0]
        index = key_parts.index(element)
        key_parts.remove(element)
        for i, part in enumerate(element.split("_")):
            key_parts.insert(index + i, part)
        return key_parts

    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        key_parts = key.split(".")
        if key.startswith("encoder.model."):
            key_parts.remove("model")
        if "stem_" in key:
            key_parts = split_underscore_element(key_parts, "stem_")
        if "stages_" in key:
            key_parts = split_underscore_element(key_parts, "stages_")
        new_key = ".".join(key_parts)
        new_state_dict[new_key] = value
    return new_state_dict


def extract_encoder(state_dict: Mapping[str, Any]) -> Dict[str, Any]:
    """Remove all entries which key does not begin with "encoder"""
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("encoder."):
            key = key[8:]
            new_state_dict[key] = value
    return new_state_dict


def network_summary(
    log_dir: Path,
    network: Any,
    logger: bool,
    encoder: str,
    distributed: bool,
    input_size: Sequence[int],
):
    depth = 4 if "nfnet" in encoder else 3
    if distributed:
        depth += 1
    net_summary = torchinfo.summary(
        network,
        input_size=input_size,
        depth=depth,
        verbose=0,
    )
    if logger:
        log.info(f"Network summary\n{net_summary}")
        with log_dir.joinpath("network.txt").open("w") as ofile:
            ofile.write(f"{net_summary}")


def tensor_reduce_average(tensor: torch.Tensor, size: int) -> torch.Tensor:
    reduced_tensor = tensor.clone()
    torch.distributed.all_reduce(reduced_tensor, op=torch.distributed.ReduceOp.SUM)
    reduced_tensor = reduced_tensor / size
    return reduced_tensor


def tensor_all_gather(tensor: torch.Tensor, size: int) -> torch.Tensor:
    result_list = [torch.ones_like(tensor) for _ in range(size)]
    torch.distributed.all_gather(result_list, tensor)
    result = torch.cat(result_list)
    return result


def logit_to_prediction(tensor: torch.Tensor) -> torch.Tensor:
    """Assumes tensor shape (N, C, H, W)"""
    return torch.nn.functional.log_softmax(tensor, dim=1).exp()


def logit_to_prediction_np(tensor: np.ndarray) -> np.ndarray:
    """Assumes tensor shape (N, C, H, W)"""
    return np.exp(scipy.special.log_softmax(tensor, axis=1))
