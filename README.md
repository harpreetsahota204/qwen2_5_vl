# Qwen2.5-VL FiftyOne Remote Model Zoo Implementation
Implementing Qwen2.5-VL as a Remote Zoo Model for FiftyOne

> NOTE: Due to recent changes in Transformers 4.50.0 (which are to be patched by Hugging Face) please ensure you have transformers<=4.49.0 installed before running the model

## Features

Based on the documentation, here's a comprehensive primer on the tasks supported by Qwen2.5-VL:

# Qwen2.5-VL Supported Tasks

1. **Visual Question Answering (VQA)**
   - Answers natural language questions about images
   - Returns text responses in English
   - Can be used for general image understanding and description

2. **Object Detection**
   - Locates and identifies objects in images
   - Returns normalized bounding box coordinates and object labels
   - Can be prompted to find specific types of objects

3. **Optical Character Recognition (OCR)**
   - Reads and extracts text from images
   - Particularly useful for documents, signs, and text-containing images

4. **Keypoint Detection**
   - Identifies specific points of interest in images
   - Returns normalized point coordinates with labels
   - Useful for pose estimation and landmark detection

5. **Image Classification**
   - Categorizes images into predefined classes
   - Can identify image quality issues
   - Returns classification labels

## Advanced Grounded Operations

The model also supports two sophisticated grounded operations that build upon VQA results:

1. **Grounded Detection**
   - Links textual descriptions with specific object locations
   - Returns detection boxes grounded in the VQA response

2. **Grounded Pointing**
   - Associates text descriptions with specific points in the image
   - Returns keypoints grounded in the VQA response

The model is highly flexible, allowing you to switch between these tasks by simply changing the operation mode and prompt, making it a versatile tool for various computer vision and multimodal applications.


## Technical Details

The model implementation:
- Supports multiple devices (CUDA, MPS, CPU)
- Uses bfloat16 precision on CUDA devices for optimal performance
- Handles various output formats including JSON parsing and coordinate normalization
- Provides comprehensive system prompts for each operation type
- Converts outputs to FiftyOne-compatible formats (Detections, Keypoints, Classifications)


## Installation

```bash
# Register the model source
foz.register_zoo_model_source("https://github.com/harpreetsahota204/qwen2_5_vl")

# Download the model
foz.download_zoo_model(
    "https://github.com/harpreetsahota204/qwen2_5_vl",
    model_name="Qwen/Qwen2.5-VL-3B-Instruct"
)
```

## Usage Examples

## Loading the model

```python
model = foz.load_zoo_model(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    # install_requirements=True #if you are using for the first time and need to download reuirement,
    # ensure_requirements=True #  ensure any requirements are installed before loading the model
)
```

#### Available Checkpoints

These checkpoints come in different sizes (3B, 7B, 32B, and 72B parameters) and each size has two variants:

- Regular version (with `-Instruct` suffix)
- AWQ quantized version (with `-Instruct-AWQ` suffix)

1. `Qwen/Qwen2.5-VL-3B-Instruct`
2. `Qwen/Qwen2.5-VL-3B-Instruct-AWQ`
3. `Qwen/Qwen2.5-VL-7B-Instruct`
4. `Qwen/Qwen2.5-VL-7B-Instruct-AWQ`
5. `Qwen/Qwen2.5-VL-32B-Instruct`
6. `Qwen/Qwen2.5-VL-32B-Instruct-AWQ`
7. `Qwen/Qwen2.5-VL-72B-Instruct`
8. `Qwen/Qwen2.5-VL-72B-Instruct-AWQ`

The AWQ versions require an additional package `autoawq==0.2.7.post3` but offer a more memory-efficient alternative to the regular versions while maintaining performance.

## Switching Between Operations

The same model instance can be used for different operations by simply changing its properties:

#### Visual Question Answering
```python

model.operation="vqa"
model.prompt="List all objects in this image seperated by commas

dataset.apply_model(model, label_field="q_vqa")
```
#### Object Detection
```python
model.operation = "detect"
model.prompt = "Locate the objects in this image."
dataset.apply_model(model, label_field="qdets")
```

#### OCR with Detection
```python
model.prompt = "Read all the text in the image."
dataset.apply_model(model, label_field="q_ocr")
```

#### Keypoint Detection
```python
model.operation = "point"
model.prompt = "Detect the keypoints in the image."
dataset.apply_model(model, label_field="qpts")
```

#### Image Classification

```python
model.operation = "classify"
model.prompt = "List the potential image quality issues in this image."
dataset.apply_model(model, label_field="q_cls")
```

### Grounded Operations
The model also supports grounded detection and pointing by using results from VQA:

```python
# Grounded Detection
dataset.apply_model(model, label_field="grounded_qdets", prompt_field="q_vqa")

# Grounded Pointing
dataset.apply_model(model, label_field="grounded_qpts", prompt_field="q_vqa")
```

Please refer to the [example notebook](using_qwen2.5-vl_as_zoo_model.ipynb) for more details 

## Output Formats

Each operation returns results in a specific format:

- **VQA (Visual Question Answering)**: Returns `str`
  - Natural language text responses in English
  
- **Detection**: Returns `fiftyone.core.labels.Detections`
  - Normalized bounding box coordinates [0,1] x [0,1]
  - Object labels
  
- **Keypoint Detection**: Returns `fiftyone.core.labels.Keypoints`
  - Normalized point coordinates [0,1] x [0,1]
  - Point labels
  
- **Classification**: Returns `fiftyone.core.labels.Classifications`
  - Class labels
  
- **Grounded Operations**: Returns same format as base operation
  - Grounded Detection: `fiftyone.core.labels.Detections`
  - Grounded Pointing: `fiftyone.core.labels.Keypoints`


## Citation

```bibtex
@article{Qwen2.5-VL,
  title={Qwen2.5-VL Technical Report},
  author={Bai, Shuai and Chen, Keqin and Liu, Xuejing and Wang, Jialin and Ge, Wenbin and Song, Sibo and Dang, Kai and Wang, Peng and Wang, Shijie and Tang, Jun and Zhong, Humen and Zhu, Yuanzhi and Yang, Mingkun and Li, Zhaohai and Wan, Jianqiang and Wang, Pengfei and Ding, Wei and Fu, Zheren and Xu, Yiheng and Ye, Jiabo and Zhang, Xi and Xie, Tianbao and Cheng, Zesen and Zhang, Hang and Yang, Zhibo and Xu, Haiyang and Lin, Junyang},
  journal={arXiv preprint arXiv:2502.13923},
  year={2025}
}
```
