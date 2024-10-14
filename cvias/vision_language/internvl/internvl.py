"""CVIAS's Vision Language Module (InternVL)."""

import logging
import re

import numpy as np
import torch
from cog_cv_abstraction.schema.detected_object import DetectedObject
from cog_cv_abstraction.vision_language._base import VisionLanguageModelBase
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from cvias.common.utils_pypkg import install_requirements
from cvias.vision_language.internvl.from_src.utility import (
    assign_device_map,
    load_image,
    load_video_from_file,
    load_video_from_seq_of_frames,
    split_model,
)

install_requirements("internvl")

MODEL_PATH = {
    "InternVL2-40B": "HuggingFace Model",
    "InternVL2-8B": "HuggingFace Model",
}


class InternVL(VisionLanguageModelBase):
    """InternVL's Vision Language Model."""

    def __init__(
        self,
        model_name: str = "InternVL2-8B",
        multi_gpus: bool = False,
        main_gpu_id_from_multi_gpus: int = 0,
    ) -> None:
        """Initialization the InternVL."""
        logging.info(
            (
                "You are using the model based on HuggingFace API.",
                "The model will be downloaded to the HuggingFace cache dir.",
            )
        )
        self.model_name = model_name
        self._path = f"OpenGVLab/{model_name}"
        self._num_gpus = torch.cuda.device_count()
        self.main_gpu_id_from_multi_gpus = main_gpu_id_from_multi_gpus
        if multi_gpus:
            device_map = split_model(
                model_name, manual_gpu_id=main_gpu_id_from_multi_gpus
            )
        else:
            device_map = assign_device_map(
                model_name=model_name, manual_gpu_id=main_gpu_id_from_multi_gpus
            )
            # main_gpu_id_from_multi_gpus = 0
        self.model = AutoModel.from_pretrained(
            self._path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map=device_map,
        ).eval()
        self.model.apply(self.move_tensors_to_gpu)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._path, trust_remote_code=True, use_fast=False
        )

    def move_tensors_to_gpu(
        self,
        module: torch.nn.Module,
    ) -> None:
        """Move all tensors in the module to GPU if they are on the CPU."""
        for name, tensor in module.named_buffers():
            if isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu":
                module.register_buffer(
                    name,
                    tensor.cuda(self.main_gpu_id_from_multi_gpus),
                    persistent=False,
                )
        for _, param in module.named_parameters():
            if param.device.type == "cpu":
                param.data = param.data.cuda(self.main_gpu_id_from_multi_gpus)

    def infer_with_image(
        self,
        language: str,
        image: np.ndarray | None = None,
        image_path: str | None = None,
        max_new_tokens: int = 1024,
        do_sample: bool = True,
    ) -> str:
        """Perform image inference with given video inputs."""
        assert (  # noqa: S101
            image is not None or image_path is not None
        ), "One of 'image' or 'image_path' must be defined."
        if image_path:
            image = Image.open(image_path).convert("RGB")
        else:
            image = Image.fromarray(image)
        # set the max number of tiles in `max_num`
        pixel_values = (
            load_image(image, max_num=12)
            .to(torch.bfloat16)
            .cuda(self.main_gpu_id_from_multi_gpus)
        )
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }

        image_prefix = "<image>\n"
        language = image_prefix + language
        # Tokenize the prompt
        # Tokenize the prompt
        inputs = self.tokenizer(language, return_tensors="pt")
        input_ids = inputs["input_ids"].to(pixel_values.device)
        attention_mask = inputs["attention_mask"].to(pixel_values.device)
        # Manually set img_context_token_id
        img_context_token = "<image>"
        img_context_token_id = self.tokenizer.encode(
            img_context_token, add_special_tokens=False
        )[0]
        self.model.img_context_token_id = img_context_token_id
        # Generate outputs with logits
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            max_new_tokens=1,  # Generate only the next token
            return_dict_in_generate=True,
            output_scores=True,  # Include scores (logits) in the output
        )
        return self.model.chat(
            self.tokenizer, pixel_values, language, generation_config
        )

    def infer_with_video(
        self,
        language: str,
        seq_of_frames: list[np.ndarray] | None = None,
        video_path: str | None = None,
        max_new_tokens: int = 1024,
        do_sample: bool = True,
        device_id: int = 0,
    ) -> str:
        """Perform image inference with given video inputs."""
        assert (  # noqa: S101
            seq_of_frames is not None or video_path is not None
        ), "One of 'seq_of_frames' or 'video_path' must be defined."
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if video_path:
            pixel_values, num_patches_list = load_video_from_file(
                video_path, device_id=device_id
            )
        else:
            pixel_values, num_patches_list = load_video_from_seq_of_frames(
                seq_of_frames=seq_of_frames, device_id=device_id
            )
        video_prefix = "".join(
            [f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))]
        )
        language = video_prefix + language
        return self.model.chat(
            self.tokenizer,
            pixel_values,
            language,
            generation_config,
            num_patches_list=num_patches_list,
            history=None,
            return_history=True,
        )

    def detect(
        self,
        frame_img: np.ndarray,
        scene_description: str,
        threshold: float = 0.10,
    ) -> DetectedObject:
        """Detect objects in the given frame image.

        Args:
            frame_img (np.ndarray): The image frame to process.
            scene_description (str): Description of the scene.
            threshold (float): Detection threshold.

        Returns:
            DetectedObject: Detected objects with their details.
        """
        parsing_rule = (
            "You must return a single float confidence value in a scale 0 to 10"
            "For example: 0.1,1.4,2.6,3.7,4.2,5.4,6.2,7.7,8.7,9.8,10.0"
            "Do not add any chatter."
            "Do not say that I cannot determine. Do your best."
        )
        prompt = (
            rf"How confidently can you say that the image describe {scene_description}."  # noqa: E501
            f"[PARSING RULE]\n:{parsing_rule}"
        )
        try:
            confidence_str = self.infer_with_image(
                language=prompt, image=frame_img
            )
            float_search = re.search(r"\d+(\.\d+)?", confidence_str)
            confidence = float(float_search.group()) if float_search else 0.10
        except SyntaxError:
            float_search = re.search(r"\d+(\.\d+)?", confidence_str)
            confidence = float(float_search.group()) if float_search else 0.10
        confidence = confidence * 1 / 10  # scale the confidence to 0-1
        confidence = min(confidence, 1.0)
        return DetectedObject(
            name=scene_description,
            model_name=self.model_name,
            confidence=round(confidence, 3),
            probability=round(confidence, 3),
            number_of_detection=1,
            is_detected=bool(confidence > threshold),
        )
