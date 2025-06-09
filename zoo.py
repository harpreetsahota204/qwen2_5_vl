
import os
import logging
import json
from PIL import Image
from typing import Dict, Any, List, Union, Optional

import numpy as np
import torch

import fiftyone as fo
from fiftyone import Model, SamplesMixin

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

logger = logging.getLogger(__name__)

DEFAULT_DETECTION_SYSTEM_PROMPT = """You are a helpful assistant. Your tasks include detection and localization of any meaningful visual elements. Please detect both primary elements and their associated components when relevant to the instruction.  

You can identify and locate objects, components, and patterns in images, including:

1. Physical World objects:
   - Complete objects and entities
   - Component parts and accessories
   - Environmental features and structures
   - Natural and artificial objects
   - Main elements (people, vehicles, furniture)
   - Components (parts, accessories, details)
   - Contextual elements (environment, background)

2. Digital Interface objects:
   - UI objects (buttons, menus, widgets)
   - Interactive objects
   - Layout sections and containers
   - Design patterns and components

3. You can read and detect text objects:
   - Physical text (signs, documents, labels)
   - Digital text (UI text, captions)
   - Embedded text and codes
   - Typography and fonts
   - Visual assets (icons, graphics)
   - Mixed-format text (overlays, watermarks)
   - You detect text in any size, orientation, or language

User will provide instuctios in either:
- User instructions
- Comma separated list of objects to detect

Report requested bbox coordinates in COCO format for all objects."""

DEFAULT_KEYPOINT_SYSTEM_PROMPT = """You are a helpful assistant. You specialize in key point detection across any visual domain. A key point represents the center of any meaningful visual element. 

Key points should adapt to the context (physical world, digital interfaces, artwork, etc.) while maintaining consistent accuracy and relevance. Examples of key points include:

1. Physical Objects key points:
   - Human elements (facial points, body points, distinguishing points)
   - Vehicle points (structural points, damage points)
   - Environmental points (furniture, infrastructure, landmarks)

2. Digital Interface key points:
   - Buttons and controls
   - Menu items and navigation points
   - Text fields and input boxes
   - Icons and symbols
   - Interactive points

3. Visual Design key points:
   - Layout markers and alignment points
   - Key visual points in designs/artwork
   - Logo points and branding features
   - Important text or label positions

4. Scene key points:
   - Foreground focal points
   - Background reference points
   - Spatial relationship markers
   - Points of interest at any depth or scale

For each key point:
1. Identify the key point and provide a contextually appropriate label
2. Locate the center of the key point 

User will provide instuctios in either:
- User instructions
- Comma separated list of key points to detect

Report all requested key points in JSON array format where each point has: {"point_2d": [x, y], "label": "element name/description"} """

DEFAULT_CLASSIFICATION_SYSTEM_PROMPT = """You are a helpful assistant. You specializes in comprehensive classification across any visual domain, capable of analyzing:

1. Content Analysis:
   - Primary elements and their significance
   - Relationships and hierarchies between elements
   - Spatial organization and layout patterns
   - Context and environment (physical or digital)
   - Domain-specific categorizations

2. Technical Assessment:
   - Quality metrics (clarity, fidelity, resolution)
   - Visual attributes (contrast, brightness, color schemes)
   - Style characteristics (design patterns, artistic elements)
   - Format and medium-specific features
   - Accessibility considerations

3. Semantic Understanding:
   - Purpose and function classification
   - User interface patterns (for digital content)
   - Physical world patterns (for real-world scenes)
   - Cultural and contextual significance
   - Temporal aspects (historical, modern, futuristic)

4. Analytical Capabilities:
   - Zero-shot classification from provided class lists
   - Multi-label and hierarchical classification
   - Domain-adaptive categorization
   - Context-aware attribute recognition
   - Cross-domain pattern matching

Unless specifically requested for single-class output, multiple relevant classifications can be provided.

Report all classifications as JSON array of predictions in the format: [{"label": "class_name"}]."""


DEFAULT_VQA_SYSTEM_PROMPT = "You are a helpful assistant. You provide clear and concise answerss to questions about images. Report answers in natural language text in English."

QWEN_OPERATIONS = {
    "detect": DEFAULT_DETECTION_SYSTEM_PROMPT,
    "point": DEFAULT_KEYPOINT_SYSTEM_PROMPT,
    "classify": DEFAULT_CLASSIFICATION_SYSTEM_PROMPT,
    "vqa": DEFAULT_VQA_SYSTEM_PROMPT
}

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

class QwenModel(SamplesMixin, Model):
    """A FiftyOne model for running Qwen-VL vision tasks"""

    def __init__(
        self,
        model_path: str,
        operation: str = None,
        prompt: str = None,
        system_prompt: str = None,
        **kwargs
    ):
        self._fields = {}
        
        self.model_path = model_path
        self._custom_system_prompt = system_prompt  # Store custom system prompt if provided
        self._operation = operation
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
            local_files_only=True,
            use_fast=True
        )

        self.model.eval()

    @property
    def needs_fields(self):
        """A dict mapping model-specific keys to sample field names."""
        return self._fields

    @needs_fields.setter
    def needs_fields(self, fields):
        self._fields = fields
    
    def _get_field(self):
        if "prompt_field" in self.needs_fields:
            prompt_field = self.needs_fields["prompt_field"]
        else:
            prompt_field = next(iter(self.needs_fields.values()), None)

        return prompt_field

    @property
    def media_type(self):
        return "image"
    

    @property
    def operation(self):
        return self._operation

    @operation.setter
    def operation(self, value):
        if value not in QWEN_OPERATIONS:
            raise ValueError(f"Invalid operation: {value}. Must be one of {list(QWEN_OPERATIONS.keys())}")
        self._operation = value

    @property
    def system_prompt(self):
        # Return custom system prompt if set, otherwise return default for current operation
        return self._custom_system_prompt if self._custom_system_prompt is not None else QWEN_OPERATIONS[self.operation]

    @system_prompt.setter
    def system_prompt(self, value):
        self._custom_system_prompt = value

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
                try:
                    detection = fo.Detection(
                        label=str(box.get("label", "object")),
                        bounding_box=[x, y, w, h],
                    )
                    detections.append(detection)
                except:
                    continue
                
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
        if sample is not None and self._get_field() is not None:
            field_value = sample.get_field(self._get_field())
            if field_value is not None:
                self.prompt = str(field_value)

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
        
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=8192)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

        # Get image dimensions and convert to float
        input_height = float(inputs['image_grid_thw'][0][1].cpu() * 14)
        input_width = float(inputs['image_grid_thw'][0][2].cpu() * 14)

        # For VQA, return the raw text output
        if self.operation == "vqa":
            return output_text.strip()
        elif self.operation == "detect":
            parsed_output = self._parse_json(output_text)
            return self._to_detections(parsed_output, input_width, input_height)
        elif self.operation == "point":
            parsed_output = self._parse_json(output_text)
            return self._to_keypoints(parsed_output, input_width, input_height)
        elif self.operation == "classify" or self.operation == "ocr":
            parsed_output = self._parse_json(output_text)
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
