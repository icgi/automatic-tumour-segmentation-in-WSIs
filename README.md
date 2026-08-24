# Automatic tumour segmentation in WSIs

Segmentation of whole-slide images (WSIs) of H&E stained histological slides into
foreground (cancerous tissue) and background (everything else).

This repository contains code used to develop and validate the method presented in the
study titled *Generalisation of automatic tumour segmentation in histopathological
whole-slide images across multiple cancer types*
([https://www.nature.com/articles/s41698-026-01311-6](https://www.nature.com/articles/s41698-026-01311-6))

## Trained model

The primary model presented in the associated study can be downloaded here: [https://zenodo.org/records/18465589](https://zenodo.org/records/18465589)

Results from applying this model on WSIs from the datasets BLCA, LUAD, LUSC, and PRAD
from The Cancer Genome Atlas (TCGA) can be accessed here: [https://zenodo.org/records/18481478](https://zenodo.org/records/18481478)

## Docker

In the project root directory, build the image defined by the `Dockerfile`.
Change the `-t <IMAGE_NAME>:<TAG>` as you like. E.g.

```
$ docker build -t name/tumour-segmentation:v01 .
```

Build an interactive container from this image.

```
$ docker run -it --gpus all --ipc host -v <SRC_PATH_1>:<DEST_PATH_1> -v <SRC_PATH_2>:<DEST_PATH_2> --name <CONTAINER_NAME> <IMAGE_NAME>:<TAG> bash
```

Once inside, you should be able to run all programs in this project.

Software dependencies with version information is provided in the supplied `Dockerfile`.

## Segmentation training

Given an input list of tiles with associated masks, one can train a model from scratch.
Code for network training and model inference is located at `process`.
There is also a `Dockerfile` specific for this purpose which is based on the NVCR
pytorch image so training should be faster.
All unneeded packages are removed compared to the root `Dockerfile`, but the included
python packages should be the same version.

Example command with distributed training using torch distributed data parallel with 1
node and 8 GPUs per node:

```
cd process
torchrun --nproc_per_node=8 src/run_training.py --config config/train.toml --input /path/to/input.csv --output /path/to/output
```

The input configuration file is the same as was used to create the published model.
The input .csv is a table with image tiles and associated masks, and is required to be on the
form

```
ImagePath,MaskPath
/path/to/image.extension,/path/to/corresponding_mask.extension
...
```

The default training configuration is set up with a DGX-A100 in mind (8 GPUs, where each GPU
has minimum 40 GB memory). Memory requirements are determined by the input size (batch
size and target tile size) and neural network architecture (encoder and decoder).

## Segmentation inference

All parts required for segmenting a WSI (preprocessing, neural network inference, and
postprocessing) are provided as runnable applications, and are combined in a script
called `full_scan_segmentation.py` for convenience.

This program excepts both a single scan, or a collection of scans either given as an
input `.csv` file or as multiple input paths.

Example:

```
$ python full_scan_segmentation.py /path/to/scans/*.svs /path/to/output /path/to/local-cache /path/to/model.tar
```

Make sure that the tile merging program (which is written in rust) is compiled. To do
so, enter `preprocess/tile_with_overlap` and run

```
cargo build --release
```

Inference with the tile size used for the published model (7680 x 7680) requires a GPU
with minimum 24 GB memory.

## Inference from c++ with libtorch

`export_traced_model.py` traces a trained model with `torch.jit` and saves it as a
TorchScript archive that can be loaded from libtorch.

```
$ python export_traced_model.py --config process/config/inference.toml \
    --restore /path/to/model.tar --output /path/to/model.pt
```

The traced module is the whole tile computation, not only the network:

- Input is a uint8 tensor `(N, H, W, 3)` with RGB channel order, as tiles are read from
  disk. Normalisation is part of the graph.
- Output is a uint8 tensor `(N, C, H, W)` with one probability map per class in the order
  given by `classes` in the config, scaled to `[0, 255]`. Softmax and 8-bit quantisation
  are part of the graph. `--class_index -1` returns `(N, 1, H, W)` with only the class
  that the merging step reads. Softmax normalises over the classes, so this reduces the
  output and the operations after it, not the cost of the network.

Nothing therefore has to be reimplemented in c++, which is what makes results agree.
The exported graph accepts any input size divisible by `min_divisor`, so a scan's edge
tiles need no separate export.

### Mixed precision

The graph is traced inside the same autocast context that `segment_images.py` uses. The
tracer records the casts that autocast inserted as ordinary operations, so the saved
graph is an explicit mixed-precision graph. Run it *without* autocast in c++.

Two settings decide whether results are reproducible, and `export_traced_model.py`
reports on both:

- Tracing requires the autocast weight cache to be off, and requires parameters not to
  require gradients. A cached cast is not recorded in the graph, and the tracer then
  either stores a stale constant or fails outright with `Cannot insert a Tensor that
  requires grad as a constant`.
- Running requires graph executor optimisation to be off. Fusing operations changes the
  order of arithmetic, and in half precision that changes the result. The optimiser only
  kicks in from the third call onwards, so a single test call will not reveal it.

```cpp
#include <torch/csrc/jit/runtime/graph_executor.h>

torch::jit::setGraphExecutorOptimize(false);
auto module = torch::jit::load("/path/to/model.pt", torch::kCUDA);
module.eval();

torch::NoGradGuard no_grad;
// tile is an OpenCV CV_8UC3 image in RGB order
auto input = torch::from_blob(tile.data, {1, tile.rows, tile.cols, 3}, torch::kUInt8)
                 .to(torch::kCUDA);
auto output = module.forward({input}).toTensor().to(torch::kCPU);
```

Do not call `torch::jit::freeze` or `optimize_for_inference` on the module. Both fold
and rewrite operations, which changes results in half precision.

The graph refers to no device, so loading it with a device and placing the input on the
same device is all that is needed to run on cpu or cuda. It cannot run on MPS, which does
not support float64: both moving the module there and the normalisation fail.

Results are identical only between builds that agree on everything that decides which
kernel runs: the libtorch version must match the pytorch version used to export (see the
`Dockerfile`), the GPU must be the same model, and the cuDNN benchmark and TF32 settings
must be left at their defaults on both sides. A traced archive does load in a later
libtorch than it was written with, but its results move, so a newer libtorch means
exporting from the matching newer pytorch.

### Differences from the python pipeline

`export_traced_model.py` measures the remaining differences against `segment_images.py`
for the same input tile and prints them. Two are known:

- One pooling layer in the timm encoder derives its padding from the input size in a way
  that ties a traced graph to the size it was traced with, and is replaced by an
  equivalent layer with fixed padding. See `replace_dynamic_padding` in
  `process/src/traced_model.py`.
- Softmax runs on the GPU rather than in scipy on the cpu, which can put the 8-bit
  probability of a pixel one level off.

Tracing prints warnings about converting tensors to python booleans and about tensors
registered as constants. Both come from size-independent code, the first from the check
on batch size inside `torch.nn.functional.batch_norm` that weight standardisation runs
into, the second from the placeholder features in `process/src/timm_nfnet_encoder.py`.
