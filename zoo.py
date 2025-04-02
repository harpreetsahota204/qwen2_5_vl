from fiftyone import Model, SamplesMixin
import torch
from PIL import Image
import logging
import json
import fiftyone as fo
import os
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from typing import Dict, Any, List, Union, Optional
import numpy as np
logger = logging.getLogger(__name__)

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEFAULT_DETECTION_SYSTEM_PROMPT = "The assistant specializes in accurate object detection and object counting. Report all detections as bounding boxes in COCO detection format."

DEFAULT_KEYPOINT_SYSTEM_PROMPT = """The assistant specializes in accurate keypoint detection and object name/description recognition. Report all keypoints JSON array where each point has format: {"point_2d": [x, y], "label": "object name/description"}"""

DEFAULT_CLASSIFICATION_SYSTEM_PROMPT = """The assistant specializes in classifying images based on the User's requirements. Unless User requests only one class, an image can have many classifications.  Report all classifications as JSON array of predictions in the format: [{label: class_name]"""

DEFAULT_VQA_SYSTEM_PROMPT = "You are an expert at answering questions about images. Provide clear and concise answers. Report results in natural language text in English."

QWEN_OPERATIONS = {
    "detect": DEFAULT_DETECTION_SYSTEM_PROMPT,
    "point": DEFAULT_KEYPOINT_SYSTEM_PROMPT,
    "classify": DEFAULT_CLASSIFICATION_SYSTEM_PROMPT,
    "vqa": DEFAULT_VQA_SYSTEM_PROMPT
}

class QwenModel(Model, SamplesMixin):
    """A FiftyOne model for running Qwen-VL vision tasks"""

    def __init__(
        self,
        model_path: str,
        operation: str = None,
        prompt: str = None,
        system_prompt: str = None,
        **kwargs
    ):
        if operation not in QWEN_OPERATIONS:
            raise ValueError(f"Invalid operation: {operation}. Must be one of {list(QWEN_OPERATIONS.keys())}")
        
        # for using sample from a dataset
        self.needs_fields = {}

        self.model_path = model_path
        
        # operation is a required parameter
        self.operation = operation
        # Use provided system prompt if given, otherwise use default
        self.system_prompt = system_prompt if system_prompt is not None else QWEN_OPERATIONS[operation]
        self.prompt = prompt
        
        self.device = get_device()
        logger.info(f"Using device: {self.device}")

        # Set dtype for CUDA devices
        self.torch_dtype = torch.bfloat16 if self.device == "cuda" else None
        # Load model and processor
        logger.info(f"Loading model from {model_path}")

        if self.torch_dtype:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                trust_remote_code=True,
                # local_files_only=True,
                device_map=self.device,
                torch_dtype=self.torch_dtype
            )
        else:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                trust_remote_code=True,
                # local_files_only=True,
                device_map=self.device,
            )
        
        logger.info("Loading processor")
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True
        )

    @property
    def media_type(self):
        return "image"

    def _parse_json(self, s: str) -> Optional[Dict]:
        """Parse JSON from model output.
        
        The model may return JSON in different formats:
        1. Raw JSON string
        2. JSON wrapped in markdown code blocks (```json ... ```)
        3. Non-JSON string (returns None)
        
        Args:
            s: String output from the model to parse
            
        Returns:
            Dict: Parsed JSON dictionary if successful
            None: If parsing fails or input is invalid
            Original input: If input is not a string
        """
        # Return input directly if not a string
        if not isinstance(s, str):
            return s
            
        # Handle JSON wrapped in markdown code blocks
        if "```json" in s:
            try:
                # Extract JSON between ```json and ``` markers
                s = s.split("```json")[1].split("```")[0].strip()
            except:
                pass
        
        # Attempt to parse the JSON string
        try:
            return json.loads(s)
        except:
            # Log first 200 chars of failed parse for debugging
            logger.debug(f"Failed to parse JSON: {s[:200]}")
            return None

    def _to_detections(self, boxes: List[Dict], image_width: int, image_height: int) -> fo.Detections:
        """Convert bounding boxes to FiftyOne Detections.
        
        Takes a list of bounding box dictionaries and converts them to FiftyOne Detection 
        objects with normalized coordinates. Handles both single boxes and lists of boxes,
        including boxes nested in dictionaries.

        Args:
            boxes: List of dictionaries or single dictionary containing bounding box info.
                Each box should have:
                - 'bbox_2d' or 'bbox': List of [x1,y1,x2,y2] coordinates in pixel space
                - 'label': Optional string label (defaults to "object")
            image_width: Width of the image in pixels
            image_height: Height of the image in pixels

        Returns:
            fo.Detections object containing the converted bounding box annotations
            with coordinates normalized to [0,1] x [0,1] range
        """
        detections = []
        
        # Handle case where boxes is a dictionary - extract list value if present
        if isinstance(boxes, dict):
            boxes = next((v for v in boxes.values() if isinstance(v, list)), boxes)
        
        # Ensure boxes is a list, even for single box input
        boxes = boxes if isinstance(boxes, list) else [boxes]
        
        for box in boxes:
            try:
                # Try to get bbox from either bbox_2d or bbox field
                bbox = box.get('bbox_2d', box.get('bbox', None))
                if not bbox:
                    continue
                    
                # Convert pixel coordinates to normalized [0,1] coordinates
                x1, y1, x2, y2 = map(float, bbox)
                x = x1 / image_width  # Normalized left x
                y = y1 / image_height # Normalized top y
                w = (x2 - x1) / image_width  # Normalized width
                h = (y2 - y1) / image_height # Normalized height
                
                # Create Detection object with normalized coordinates
                detection = fo.Detection(
                    label=str(box.get("label", "object")),
                    bounding_box=[x, y, w, h],
                )
                detections.append(detection)
                
            except Exception as e:
                # Log any errors processing individual boxes but continue
                logger.debug(f"Error processing box {box}: {e}")
                continue
                
        return fo.Detections(detections=detections)

    def _to_keypoints(self, points: List[Dict], image_width: int, image_height: int) -> fo.Keypoints:
        """Convert a list of point dictionaries to FiftyOne Keypoints.
        
        Args:
            points: List of dictionaries containing point information.
                Each point should have:
                - 'point_2d': List of [x,y] coordinates in pixel space
                - 'label': String label describing the point
            image_width: Width of the image in pixels
            image_height: Height of the image in pixels
                
        Returns:
            fo.Keypoints object containing the converted keypoint annotations
            with coordinates normalized to [0,1] x [0,1] range
        
        Expected input format:
        [
            {"point_2d": [100, 200], "label": "person's head", "confidence": 0.9},
            {"point_2d": [300, 400], "label": "dog's nose"}
        ]
        """
        keypoints = []
        
        for point in points:
            try:
                # Get coordinates from point_2d field and convert to float
                x, y = point["point_2d"]
                x = float(x.cpu() if torch.is_tensor(x) else x)
                y = float(y.cpu() if torch.is_tensor(y) else y)
                
                normalized_point = [
                    x / image_width,
                    y / image_height
                ]
                
                keypoint = fo.Keypoint(
                    label=str(point.get("label", "point")),
                    points=[normalized_point],
                )
                keypoints.append(keypoint)
            except Exception as e:
                logger.debug(f"Error processing point {point}: {e}")
                continue
                
        return fo.Keypoints(keypoints=keypoints)

    def _to_classifications(self, classes: List[Dict]) -> fo.Classifications:
        """Convert a list of classification dictionaries to FiftyOne Classifications.
        
        Args:
            classes: List of dictionaries containing classification information.
                Each dictionary should have:
                - 'label': String class label
                
        Returns:
            fo.Classifications object containing the converted classification 
            annotations with labels and optional confidence scores
            
        Example input:
            [
                {"label": "cat",},
                {"label": "dog"}
            ]
        """
        classifications = []
        
        # Process each classification dictionary
        for cls in classes:
            try:
                # Create Classification object with required label and optional confidence
                classification = fo.Classification(
                    label=str(cls["label"]),  # Convert label to string for consistency
                )
                classifications.append(classification)
            except Exception as e:
                # Log any errors but continue processing remaining classifications
                logger.debug(f"Error processing classification {cls}: {e}")
                continue
                
        # Return Classifications container with all processed results
        return fo.Classifications(classifications=classifications)

    def _predict(self, image: Image.Image, sample=None) -> Union[fo.Detections, fo.Keypoints, fo.Classifications, str]:
        """Process a single image through the model and return predictions.
        
        This internal method handles the core prediction logic including:
        - Constructing the chat messages with system prompt and user query
        - Processing the image and text through the model
        - Parsing the output based on the operation type (detection/points/classification/VQA)
        
        Args:
            image: PIL Image to process
            sample: Optional FiftyOne sample containing the image filepath
            
        Returns:
            One of:
            - fo.Detections: For object detection results
            - fo.Keypoints: For keypoint detection results  
            - fo.Classifications: For classification results
            - str: For VQA text responses
            
        Raises:
            ValueError: If no prompt has been set
        """
        if self.prompt is None:
            raise ValueError("A prompt must be provided before prediction")

        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.prompt},
                    {"image": sample.filepath if sample else None}
                ]
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt").to(self.device)
        
        output_ids = self.model.generate(**inputs, max_new_tokens=4096)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

        # Get image dimensions and convert to float
        input_height = float(inputs['image_grid_thw'][0][1].cpu() * 14)
        input_width = float(inputs['image_grid_thw'][0][2].cpu() * 14)

        # For VQA, return the raw text output
        if self.operation == "vqa":
            return output_text.strip()

        # For other operations, parse JSON and convert to appropriate format
        parsed_output = self._parse_json(output_text)
        if not parsed_output:
            return None

        if self.operation == "detect":
            return self._to_detections(parsed_output, input_width, input_height)
        elif self.operation == "point":
            return self._to_keypoints(parsed_output, input_width, input_height)
        elif self.operation == "classify":
            return self._to_classifications(parsed_output)

    def predict(self, image, sample=None):
        """Process an image with the model.
        
        A convenience wrapper around _predict that handles numpy array inputs
        by converting them to PIL Images first.
        
        Args:
            image: PIL Image or numpy array to process
            sample: Optional FiftyOne sample containing the image filepath
            
        Returns:
            Model predictions in the appropriate format for the current operation
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return self._predict(image, sample)
