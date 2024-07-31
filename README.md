# Barcode OCR (Character Recognition only) | CRNN (CNN - RNN - CTC loss) | PyTorch, Lightning, ClearML

<a href="https://clear.ml/docs/latest/"><img alt="ClearML" src="https://img.shields.io/badge/MLOps-Clear%7CML-%2309173c"></a>

This repo is capable of training, validation and export to ONNX of a configurable RCNN PyTorch model using Lightning and ClearML to perform barcodes character recognition task.

**The project features:**

1. PyTorch implementation of RCNN character recognition architecture applied to the task of barcode character recognition.
1. Notebook that helps to configure the architecture with given dataset.
1. Full-scale experiment tracking in [CLearML](https://clear.ml/) including:
   1. Config and hyperparameter tracking
   1. Model artifacts versioning (`.pt`, `.onnx`)
   1. Tracking of numerous losses, metrics (see below) and visualizations of batches and model predicts.
1. Extensively configurable end-to-end RCNN model training and evaluation pipeline built on [Lightning](https://lightning.ai/docs/pytorch/stable/) that:
   1. Has a clean and easy configuration in a single `yaml` config file.
   1. Optionally pulls the dataset from ClearML.
   1. Builds RCNN model according to config (backbone and RNN architecture and parameters), downloading pretrained CNN backbone from [timm](https://github.com/huggingface/pytorch-image-models)
   1. Trains the model to recognize barcode characters.
   1. Finds the best checkpoint (epoch) based on string match metric on validation set.
   1. Evaluates string match and edit distance metrics on both validation and test sets.
   1. Exports the best checkpoint to [ONNX](https://onnx.ai/).
1. Barcode OCR dataset preprocessing pipeline, featuring data versioning in ClearML.
1. CI that runs static code analysis configured in `pre-commit.yaml` (tests are WIP).

Check out the example of RCNN barcode OCR model training experiment in ClearML.

______________________________________________________________________

## Getting started

1. Follow [instructions](https://github.com/python-poetry/install.python-poetry.org) to install Poetry. Check that poetry was installed successfully:
   ```bash
   poetry --version
   ```
1. Setup workspace.
   - Unix:
   ```bash
   make setup_ws
   ```
   - Windows:
   ```bash
   make setup_ws PYTHON_EXEC=<path_to_your_python_3.10_executable>
   ```
1. Activate poetry virtual environment
   ```bash
   poetry shell
   ```

______________________________________________________________________

# Model Training

## I. Barcode OCR

### 1. Configure (or skip to use defaults, which is perfectly fine)

1. [Data config](configs/data.yaml) (`configs/data.yaml`) to set how to split the data and whether to version it in ClearML.
1. [Train and evaluation config](configs/train_eval.yaml) (`configs/train_eval.yaml`) to set everything else: use local dataset or from ClearML, which pretrain to use, hyperparameters, ClearML tracking settings, etc.

### 2.  Run data pipeline

To download and preprocess barcodes detection dataset

```bash
make run_data_pipe
```

OR the same thing in 2 steps:

```bash
make fetch_data
make prep_data
```

### 3. Run training and evaluation pipeline

```bash
make run_train_eval
```

That's it, RCNN goes brrr, and you'll be able to see all the results and your model in ClearML, already trained and exported to ONNX.

### 4. \[Alternatively\] Run end-to-end pipeline

To run everything at once in a single line: fetch data + preprocess data + train and evaluate pretrained RT DETR model.

```bash
make run_e2e_pipeline
```

______________________________________________________________________

# Acknowledgments

1. Firstly, thanks a lot to the authors of the amazing RT DETR architecture, that is so fast and well-performing that I managed to train a decent model on CPU in just a few hours using a small dataset.
1. Secondly, shoutout to the team of [DeepShchool's](https://deepschool.ru/) [CV Rocket course](https://deepschool.ru/cvrocket), where I (hopefully) learned best practices of ML development and ClearML experiment tracking. BTW this is one of the course's graduation projects :)

______________________________________________________________________

# TODO List

1. Add sanity and some unit tests.
1. Add conversion of the model to [OpenVino](https://docs.openvino.ai/2024/index.html) to the pipeline.
