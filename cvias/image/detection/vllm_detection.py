"""VLLM Detection Model."""

import base64
import logging

import cv2
import numpy as np
from cog_cv_abstraction.schema.detected_object import DetectedObject
from neus_v.calibration.vlm import smooth_mapping
from openai import OpenAI

from cvias.image.detection import CviasDetectionModel


class VLLMDetection(CviasDetectionModel):
    """VLLM Detection Model."""

    def __init__(
        self,
        api_key: str = "EMPTY",
        api_base: str = "http://localhost:8000/v1",
        model: str = "OpenGVLab/InternVL2_5-8B",
        calibration_method: str | None = None,
    ) -> None:
        """Initialize VLLM Detection Model.

        Args:
            api_key (str): The API key for the VLLM Detection Model.
            api_base (str): The API base for the VLLM Detection Model.
            model (str): The model for the VLLM Detection Model.
            calibration_method (str | None): The calibration method for the VLLM Detection Model.
        """  # noqa: E501
        super().__init__(calibration_method=calibration_method)
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.model = model
        self.system_message = (
            "You must only return a Yes or No, and not both, to any question asked.\n"  # noqa: E501
            "You must not include any other symbols, information, text, justification in your answer or repeat Yes or No multiple times.\n"  # noqa: E501
            "For example, if the question is 'Is there a cat present in the Imag    e?', the answer must only be 'Yes' or 'No'."  # noqa: E501
        )

    def _encode_frame(self, frame: np.ndarray) -> str:
        """Encode a uint8 numpy array (image) as a JPEG and then base64 encode it.

        Args:
            frame (np.ndarray): The frame image to encode.

        Returns:
            str: The base64 encoded frame image.
        """  # noqa: E501
        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            msg = "Could not encode frame"
            logging.error(msg)
            raise ValueError(msg)
        return base64.b64encode(buffer).decode("utf-8")

    def detect(
        self,
        frame_img: np.ndarray | None = None,
        classes: list[np.ndarray] | None = None,
        threshold: float | None = None,
    ) -> DetectedObject:
        """Detect the scene description in the frame or sequence of frames.

        Args:
            frame_img (np.ndarray | None): The frame image to detect.
            classes (list[np.ndarray] | None): The classes to detect.
            threshold (float | None): The threshold to use for the detection.

        Returns:
            DetectedObject: The detected object.
        """
        seq_of_frames = [frame_img]
        scene_description = classes[0]

        # Encode each frame.
        encoded_images = [self._encode_frame(frame) for frame in seq_of_frames]

        # Build the user message: a text prompt plus one image for each frame.
        user_content = [
            {
                "type": "text",
                "text": f"Does the sequence of these images depict '{scene_description}'",  # noqa: E501
            }
        ]
        for encoded in encoded_images:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                }
            )

        # Create a chat completion request.
        chat_response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": user_content},
            ],
            max_tokens=1,
            temperature=0.0,
            logprobs=True,
            top_logprobs=20,
        )

        # Retrieve the list of TopLogprob objects.
        top_logprobs_list = (
            chat_response.choices[0].logprobs.content[0].top_logprobs
        )

        # Build a mapping from token text (stripped) to its probability.
        token_prob_map = {}
        for top_logprob in top_logprobs_list:
            token_text = top_logprob.token.strip()
            token_prob_map[token_text] = np.exp(top_logprob.logprob)

        # Extract probabilities for "Yes" and "No"
        yes_prob = token_prob_map.get("Yes", 0.0)
        no_prob = token_prob_map.get("No", 0.0)

        # Compute the normalized probability for "Yes": p_yes / (p_yes + p_no)
        if yes_prob + no_prob > 0:
            confidence = yes_prob / (yes_prob + no_prob)
        else:
            msg = "No probabilities for 'Yes' or 'No' found in the response."
            logging.error(msg)
            raise ValueError(msg)

        if threshold:
            confidence = smooth_mapping(
                confidence=confidence, false_threshold=threshold
            )
            if confidence < threshold:
                no_prob = 1.0

        detected_object = DetectedObject(
            name=scene_description,
            model_name=self.model,
            confidence_of_all_obj=[round(confidence, 3)],
            probability_of_all_obj=[],
            number_of_detection=1,
            is_detected=yes_prob > no_prob,  # TODO: Check if this is correct
        )
        if self.calibration_method:
            # calibrate confidence score
            if "internvl2" not in self.model.lower():
                logging.warning(
                    "Temperature scaling calibration is only supported for InternVL models."  # noqa: E501
                )
                return self.no_calibration(detected_object)
            return self.calibrate(detected_object)

        return self.no_calibration(detected_object)

    def no_calibration(self, detected_object: DetectedObject) -> DetectedObject:
        """No calibration.

        Args:
            detected_object (DetectedObject): Detected object.
        """
        probabilities = []
        if detected_object.is_detected:
            for confidence in detected_object.confidence_of_all_obj:
                probabilities.append(confidence)
            detected_object.probability_of_all_obj = probabilities
            detected_object.probability = max(probabilities)
        return detected_object

    def calibrate(self, detected_object: DetectedObject) -> DetectedObject:
        """Calibrate detection results.

        Args:
            detected_object (DetectedObject): Detected object.

        Returns:
            DetectedObject: Calibrated detected object.
        """
        probabilities = []
        if detected_object.is_detected:
            for confidence in detected_object.confidence_of_all_obj:
                probabilities.append(
                    self.calibrate_confidence(
                        confidence=confidence,
                        calibration_func=self.calibrate_function,
                    )
                )
            detected_object.probability_of_all_obj = probabilities
            detected_object.probability = max(probabilities)
        return detected_object

    def calibrate_function(
        self,
        confidence: float,
        true_threshold: float = 0.95,
        false_threshold: float = 0.40,
        target_conf: float = 0.60,
        target_prob: float = 0.78,
        p_min: float = 0.01,
        p_max: float = 0.99,
        steepness_factor: float = 0.7,  # New parameter: 0-1 range, lower = less steep # noqa: E501
    ) -> float:  # no[]
        """Map confidence to probability using a sigmoid function with adjustable steepness.

        Args:
            confidence: Input confidence score
            true_threshold: Upper threshold (0.78)
            false_threshold: Lower threshold (0.40)
            target_conf: Target confidence point (0.60)
            target_prob: Target probability value (0.78)
            p_min: Minimum probability (0.01)
            p_max: Maximum probability (0.99)
            steepness_factor: Controls curve steepness (0-1, lower = less steep)
        """  # noqa: E501
        if confidence <= false_threshold:
            return p_min

        if confidence >= true_threshold:
            return p_max

        # Calculate parameters to ensure target_conf maps to target_prob
        # For a sigmoid function: f(x) = L / (1 + e^(-k(x-x0)))

        # First, normalize the target point
        x_norm = (target_conf - false_threshold) / (
            true_threshold - false_threshold
        )
        y_norm = (target_prob - p_min) / (p_max - p_min)

        # Find x0 (midpoint) and k (steepness) to satisfy our target point
        x0 = 0.30  # Midpoint of normalized range

        # Calculate base k value to hit the target point
        base_k = -np.log(1 / y_norm - 1) / (x_norm - x0)

        # Apply steepness factor (lower = less steep)
        k = base_k * steepness_factor

        # With reduced steepness, we need to adjust x0 to still hit the target point # noqa: E501
        # Solve for new x0: y = 1/(1+e^(-k(x-x0))) => x0 = x + ln(1/y-1)/k
        adjusted_x0 = x_norm + np.log(1 / y_norm - 1) / k

        # Apply the sigmoid with our calculated parameters
        x_scaled = (confidence - false_threshold) / (
            true_threshold - false_threshold
        )
        sigmoid_value = 1 / (1 + np.exp(-k * (x_scaled - adjusted_x0)))

        # Ensure we still hit exactly p_min and p_max at the thresholds
        # by rescaling the output slightly
        min_val = 1 / (1 + np.exp(-k * (0 - adjusted_x0)))
        max_val = 1 / (1 + np.exp(-k * (1 - adjusted_x0)))

        # Normalize the output
        normalized = (sigmoid_value - min_val) / (max_val - min_val)

        return p_min + normalized * (p_max - p_min)
