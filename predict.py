from typing import List
from cog import BasePredictor, Input, Path
from transformers import MllamaForConditionalGeneration, MllamaProcessor

import torch
from PIL import Image
import requests

CACHE_DIR = 'weights'
MODEL_ID = "unsloth/Llama-3.2-90B-Vision-Instruct"

class Predictor(BasePredictor):
    def setup(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = MllamaForConditionalGeneration.from_pretrained(
            MODEL_ID,
            cache_dir=CACHE_DIR,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = MllamaProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)

    def predict(
        self,
        image: Path = Input(description="Input image"),
        prompt: str = Input(description="Input text for the model"),
        temperature: float = Input(description="Controls randomness. Lower values make the model more deterministic, higher values make it more random.", default=0.7, ge=0.0, le=1.0),
        top_p: float = Input(description="Controls diversity of the output. Lower values make the output more focused, higher values make it more diverse.", default=0.95, ge=0.0, le=1.0),
        max_new_tokens: int = Input(
            description="Maximum number of tokens to generate. A word is generally 2-3 tokens",
            ge=1,
            default=512
        ),    ) -> List[str]:

        image = Image.open(image).convert('RGB')
        

        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]}
        ]
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        inputs = self.processor(image, input_text, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, temperature=temperature, top_p=top_p, max_new_tokens=max_new_tokens)

        
        return [self.processor.decode(output, skip_special_tokens=True) for output in outputs]
