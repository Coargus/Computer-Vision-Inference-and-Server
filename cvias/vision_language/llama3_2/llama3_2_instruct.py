"""CVIAS's Vision Language Module (InternVL)."""

import gc
import logging
import re

import numpy as np
import torch
from cog_cv_abstraction.schema.detected_object import DetectedObject
from cog_cv_abstraction.vision_language._base import VisionLanguageModelBase
from PIL import Image
from torch.nn.functional import softmax
from transformers import (
    AutoProcessor,
    MllamaForConditionalGeneration,
)

from cvias.common.utils_pypkg import install_requirements

install_requirements("llama3_2")

MODEL_PATH = {
    "Llama-3.2-11B-Vision-Instruct": "HuggingFace Model",
    "Llama-3.2-90B-Vision-Instruct": "HuggingFace Model",
}


class Llama32VisionInstruct(VisionLanguageModelBase):
    """InternVL's Vision Language Model."""

    def __init__(
        self,
        model_name: str = "Llama-3.2-11B-Vision-Instruct",
        device: int = 0,
        multi_gpus: bool = False,
    ) -> None:
        """Initialization the InternVL."""
        logging.info(
            (
                "You are using the model based on HuggingFace API.",
                "The model will be downloaded to the HuggingFace cache dir.",
            )
        )
        self.model_name = model_name
        self._path = f"meta-llama/{model_name}"
        device_map = {"": f"cuda:{device}"}

        if multi_gpus:
            device_map = "auto"

        self.model = MllamaForConditionalGeneration.from_pretrained(
            self._path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
        )
        self.processor = AutoProcessor.from_pretrained(self._path)
        self.device = self.model.device
        if multi_gpus:
            self.device = next(self.model.modules()).device

    def clear_gpu_memory(self) -> None:
        """Clear CUDA cache and run garbage collection to free GPU memory."""
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()
        gc.collect()  # Run garbage collector

    def post_process_output(
        self, output: str, return_index: bool = False
    ) -> str:
        """Post processing the output."""
        start = "<|start_header_id|>assistant<|end_header_id|>"
        end = "<|eot_id|>"
        start_idx = output.find(start)
        if start_idx == -1:
            return output
        end_idx = output.find(end, start_idx + len(start))
        if end_idx == -1:
            return output[start_idx + len(start) :]

        response_index = (start_idx + len(start) + 2, end_idx)
        if return_index:
            return response_index
        return output[response_index[0] : response_index[-1]]

    def infer_with_image(
        self,
        language: str,
        image: np.ndarray | None = None,
        image_path: str | None = None,
        max_new_tokens: int = 1024,
        add_generation_prompt: bool = True,
    ) -> str:
        """Perform image inference with given video inputs."""
        assert (  # noqa: S101
            image is not None or image_path is not None
        ), "One of 'image' or 'image_path' must be defined."
        if image_path:
            image = Image.open(image_path).convert("RGB")
        else:
            image = Image.fromarray(image)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": language,
                    },
                ],
            }
        ]
        input_text = self.processor.apply_chat_template(
            messages, add_generation_prompt=add_generation_prompt
        )
        inputs = self.processor(
            image, input_text, add_special_tokens=False, return_tensors="pt"
        ).to(self.device)

        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
        )
        return self.post_process_output(self.processor.decode(output[0]))

    def infer_with_image_confidence(
        self,
        language: str,
        image: np.ndarray | None = None,
        image_path: str | None = None,
        max_new_tokens: int = 1024,
        add_generation_prompt: bool = True,
    ) -> str:
        """Perform image inference with given video inputs."""
        assert (  # noqa: S101
            image is not None or image_path is not None
        ), "One of 'image' or 'image_path' must be defined."
        if image_path:
            image = Image.open(image_path).convert("RGB")
        else:
            image = Image.fromarray(image)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": language,
                    },
                ],
            }
        ]
        input_text = self.processor.apply_chat_template(
            messages, add_generation_prompt=add_generation_prompt
        )
        inputs = self.processor(
            image, input_text, add_special_tokens=False, return_tensors="pt"
        ).to(self.device)

        generation_config = {
            "output_scores": True,
            "output_logits": True,
            "return_dict_in_generate": True,
        }
        output = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, **generation_config
        )
        response = self.post_process_output(
            self.processor.decode(output.sequences[0])
        )

        generated_ids = output.sequences[0].tolist()
        assistant_index = generated_ids.index(78191)
        start_index = (
            generated_ids.index(128007, assistant_index) + 1
        )  # +1 since answer starts after \n\n
        end_index = generated_ids.index(128009, assistant_index)

        token_to_compute = (
            output.sequences[0]
            .detach()
            .cpu()
            .numpy()[start_index + 1 : end_index]
        )

        confidence = 1.0
        for logit_idx in range(1, len(output.logits) - 1):
            prob = softmax(output.logits[logit_idx])[
                0, token_to_compute[logit_idx]
            ]
            confidence = prob.item() * confidence

        return response, confidence

    def detect(
        self,
        frame_img: np.ndarray,
        scene_description: str,
        threshold: float = 0.1,
        confidence_as_token_probability: bool = False,
    ) -> DetectedObject:
        """Detect objects in the given frame image.

        Args:
            frame_img (np.ndarray): The image frame to process.
            scene_description (str): Description of the scene.
            threshold (float): Detection threshold.
            confidence_as_token_probability (bool):
                Whether to use token probability as confidence.

        Returns:
            DetectedObject: Detected objects with their details.
        """
        if confidence_as_token_probability:
            parsing_rule = [
                "You must only return a Yes or No, and not both, to any question asked. "  # noqa: E501
                "You must not include any other symbols, information, text, justification in your answer or repeat Yes or No multiple times.",  # noqa: E501
                "For example, if the question is 'Is there a cat present in the Image?', the answer must only be 'Yes' or 'No'.",  # noqa: E501
            ]
            parsing_rule = "\n".join(parsing_rule)
            prompt = (
                rf"Is there a {scene_description} present in the image? "
                f"[PARSING RULE]\n:{parsing_rule}"
            )

            response, confidence = self.infer_with_image_confidence(
                language=prompt, image=frame_img
            )
            # TODO: Add a check for the response to be Yes or NO or clean up response better  # noqa: E501
            if "yes" in response.lower():
                detected = True
                if confidence <= threshold:
                    confidence = 0.0
                    detected = False
                probability = confidence
            else:
                detected = False
            probability = confidence
        else:
            parsing_rule = (
                "You must return a single float confidence value in a scale 0 to 10"  # noqa: E501
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
                confidence = (
                    float(float_search.group()) if float_search else 0.10
                )
            except SyntaxError:
                float_search = re.search(r"\d+(\.\d+)?", confidence_str)
                confidence = (
                    float(float_search.group()) if float_search else 0.10
                )
            confidence = confidence * 1 / 10  # scale the confidence to 0-1
            confidence = min(confidence, 1.0)
            probability = confidence
            detected = True
            if confidence <= threshold:
                detected = False

        self.clear_gpu_memory()

        return DetectedObject(
            name=scene_description,
            model_name=self.model_name,
            confidence=round(confidence, 3),
            probability=round(probability, 3),
            number_of_detection=1,
            is_detected=detected,
        )

    def infer_with_video(self) -> None:
        """Perform video inference (not implemented)."""
        msg = "Not implemented yet."
        raise NotImplementedError(msg)
