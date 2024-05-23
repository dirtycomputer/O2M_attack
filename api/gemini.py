from PIL import Image
import google.generativeai as genai
from google.generativeai import GenerationConfig

from apikeys import apikeys

genai.configure(api_key=apikeys.gemini)


def gemini(image_path: str, text: str, model: str = 'gemini-pro-vision', max_tokens: int = 300):
    img = Image.open(image_path)
    model = genai.GenerativeModel(model)
    response = model.generate_content(
        [text, img], 
        stream=True,
        generation_config=GenerationConfig(max_output_tokens=max_tokens)
        )
    response.resolve()
    return response.text