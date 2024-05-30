import torch
from PIL import Image
import os
import pandas as pd
from transformers import CLIPProcessor, CLIPModel
import time

def time_decorator(func):
    log_file = 'function_times.log'  # 日志文件
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        with open(log_file, 'a') as f:
            f.write(f"Function {func.__name__} ran in {end_time - start_time} seconds\n")
        return result
    return wrapper

class ImageTextSimilarity:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
        self.processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
        self.model.to(self.device)
        self.model.eval()
        print("openai model loaded successfully.")

    @time_decorator
    def get_clip_score(self, image_path, text):
        # Set a fixed max length for text sequences in the CLIP model
        max_length = 77  # This is the typical max sequence length for text in the CLIP model

        # Truncate or pad the text to this maximum length
        text = text[:max_length]

        image = Image.open(image_path)
        inputs = self.processor(text=text, images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        return logits_per_image.item()

    def calculate_similarity(self, image_path, text_folder):
        results = []
        if not os.path.exists(image_path):
            raise FileNotFoundError("The image file does not exist.")
        if not os.path.isdir(text_folder):
            raise ValueError("Provided text_folder is not a directory.")
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        text_path = os.path.join(text_folder, base_name + '.txt')
        if os.path.exists(text_path):
            with open(text_path, 'r') as file:
                text = file.read()
            score = self.get_clip_score(image_path, text)
            results.append((image_path, text_path, score))
        else:
            print(f"Text file not found for image {image_path}")
        return pd.DataFrame(results, columns=['Image', 'Text File', 'Similarity Score'])

    @time_decorator
    def get_clip_score_path(self, image_path, text):
        # Truncate or pad the text to a maximum length of 77 (typical for CLIP model)
        max_length = 77
        text = text[:max_length]

        # Prepare image and text inputs for the model
        image = Image.open(image_path)
        inputs = self.processor(text=text, images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits_per_image = outputs.logits_per_image
        return logits_per_image.item()

    def calculate_similarity_path(self, image_path, text):
        if not os.path.exists(image_path):
            raise FileNotFoundError("The image file does not exist.")
        score = self.get_clip_score_path(image_path, text)
        return pd.DataFrame({
            'Image': [image_path],
            'Text': [text],
            'Similarity Score': [score]
        })

    @time_decorator
    def get_clip_score_hf(self, image, text):
        # Set a fixed max length for text sequences in the CLIP model
        max_length = 77  # This is the typical max sequence length for text in the CLIP model

        # Truncate or pad the text to this maximum length
        text = text[:max_length]

        # Check if the image is already a PIL.Image object
        if not isinstance(image, Image.Image):
            # If it's a path, open it
            image = Image.open(image)

        inputs = self.processor(text=text, images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        return logits_per_image.item()

    def calculate_similarity_hf(self, image, text):
        if not isinstance(image, Image.Image) and not os.path.exists(image):
            raise FileNotFoundError("The image file does not exist or image is not an Image object.")

        score = self.get_clip_score_hf(image, text)
        return pd.DataFrame({
            'Image': ['In-memory image' if isinstance(image, Image.Image) else image],
            'Text': [text],
            'Similarity Score': [score]
        })

# # Below is how to use the updated function
# if __name__ == '__main__':
#     similarity_calculator_i2t = ImageTextSimilarity()
#     normal_text_folder = './output_normal'
#     harmful_text_folder = './output_harmful'
#     image_folder = './images'

#     def calculate_and_print_image_text_similarity(image_folder, normal_text_folder, harmful_text_folder):
#         for image_file in os.listdir(image_folder):
#             if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
#                 image_path = os.path.join(image_folder, image_file)
#                 results_normal = similarity_calculator_i2t.calculate_similarity(image_path, normal_text_folder)
#                 results_harmful = similarity_calculator_i2t.calculate_similarity(image_path, harmful_text_folder)
                
#                 print(f"Results for {image_path} with normal texts:")
#                 print(results_normal)
#                 print(f"Results for {image_path} with harmful texts:")
#                 print(results_harmful)

#     # Execute the function
#     calculate_and_print_image_text_similarity(image_folder, normal_text_folder, harmful_text_folder)
