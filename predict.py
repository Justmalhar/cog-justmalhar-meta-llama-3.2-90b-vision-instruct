# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

model_id = "unsloth/Llama-3.2-90B-Vision-Instruct"


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""      
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'  
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

    def predict(
        self,
       image: Path = Input(description="Input image"),
        prompt: str = Input(description="Text prompt to send to the model."),
        max_new_tokens: int = Input(description="Maximum number of tokens to generate. A word is generally 2-3 tokens.")
    ) -> Path:
        """Run a single prediction on the model"""
        image = Image.open(image).convert('RGB')

        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]}
        ]
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(model.device)

        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return [self.processor.decode(output, skip_special_tokens=True) for output in outputs]

