# Qwen2.5-VL FiftyOne Remote Model Zoo Implementation
Implementing Qwen2.5-VL as a Remote Zoo Model for FiftyOne

> NOTE: Due to recent changes in Transformers 4.50.0 (which are to be patched by Hugging Face) please ensure you have transformers<=4.49.0 installed before running the model

## Features

The model supports multiple vision-language operations:
- Visual Question Answering (VQA)
- Object Detection
- OCR with Detection
- Keypoint Detection (Pointing)
- Image Classification
- Grounded Detection and Pointing

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

### Visual Question Answering
```python
model = foz.load_zoo_model(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    operation="vqa",
    prompt="List all objects in this image seperated by commas"
)
dataset.apply_model(model, label_field="q_vqa")
```

### Object Detection
```python
model.operation = "detect"
model.prompt = "Locate the objects in this image."
dataset.apply_model(model, label_field="qdets")
```

### OCR with Detection
```python
model.prompt = "Read all the text in the image."
dataset.apply_model(model, label_field="q_ocr")
```

### Keypoint Detection
```python
model.operation = "point"
model.prompt = "Detect the keypoints in the image."
dataset.apply_model(model, label_field="qpts")
```

### Image Classification
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
