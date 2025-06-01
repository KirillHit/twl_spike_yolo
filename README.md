# TWL Spike Yolo

**TWL (Time Window based Learning) Spike Yolo** is a spiking neural network (SNN) for object detection with event-based cameras. The model is built on the YOLOv8 architecture, adapted to process event streams. This approach enables continuous data processing with low latency and high energy efficiency. The project also explores the use of multimodal data, combining event-based and frame-based inputs to improve detection performance in challenging scenarios.

<p align="center">
  <img src="https://github.com/user-attachments/assets/4a04ffaa-8e44-4329-a754-4e59dcf894d7" alt="gif"><br>
  <em>Demonstration of model performance on the Gen1 dataset</em>
</p>

## Citation

A publication with a detailed explanation is planned and will be available soon.  
A link to the article will be added here after publication.

## Requirements

All dependencies are provided in the `environment.yml` file. Main dependencies: [PyTorch](https://pytorch.org/), [Norse](https://github.com/norse/norse), [Lightning](https://lightning.ai/), [Faster-COCO-Eval](https://github.com/MiXaiLL76/faster_coco_eval)

## Datasets

This project uses two datasets for event-based object detection:

- [**Gen1**]((https://www.prophesee.ai/2020/01/24/prophesee-gen1-automotive-detection-dataset/))  
  Large-scale event-based detection dataset collected with the ATIS sensor in real driving scenarios. Contains annotated events with timestamps and bounding boxes for cars and pedestrians.

- [**DSEC-detection**](https://dsec.ifi.uzh.ch/dsec-detection/)  
  Extension of the DSEC dataset for detection tasks, collected with the DAVIS346 sensor. Provides synchronized events and RGB images with bounding box annotations for multiple object classes in diverse real-world driving scenarios. Used in this project to explore multimodal (event + frame) object detection approaches.

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/KirillHit/twl_spike_yolo.git
   cd twl_spike_yolo
   ```

2. **Create and activate the environment**
   ```bash
   conda env create -f environment.yml
   conda activate twl_spike_yolo
   ```

3. **Prepare datasets**  
   Download and extract the required dataset(s). By default, the dataset path is assumed to be `./data`. You can override this by specifying the path in the config file under the `data.root` parameter, or by passing it at runtime as `--data.root={dataset_folder_path}`.

4. **Train, validate, test, or run prediction with a specific config**
   ```bash
   python3 main.py {fit/val/test/predict} --config config/{model_name}.yaml
   ```
   Replace `{fit/val/test/predict}` with the desired stage and `{model_name}` with your config name.  
   Here, `model_name` should be the name of one of the provided configuration files in the `config` folder (such as `yolo8l_gen1`, `yolo8l_dsec_cnn`, etc.) or your own config file.  
   Model weights are downloaded automatically from [Hugging Face](https://huggingface.co/KirillHit/twl_spike_yolo/tree/main) if not present locally.

## Model evaluation and analysis tools

The `tools` folder contains scripts for evaluating and analyzing models:

- **estimate_energy.py** — estimates the energy efficiency of a model using the provided config.
- **estimate_activity.py** — analyzes neuron activity in the network layers.
- **estimate_runtime_map.py** — evaluates model performance on continuous data streams.

Example usage for energy estimation:
```bash
python3 -m tools.estimate_energy --config config/{model_name}.yaml
```

The example below shows neuron activity visualization obtained using `estimate_activity.py`:
```bash
python3 -m tools.estimate_activity --config config/{model_name}.yaml
```

<p align="center">
  <img src="https://github.com/user-attachments/assets/a5c85d28-880f-48f9-9ae6-14522c94e1f4" width="500"><br>
  <em>Demonstration of neuron activity in one of the network layers</em>
</p>

## Results

Results and evaluation metrics will be described here after publication.

