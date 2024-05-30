import torch
from PIL import Image
import open_clip
from urllib.request import urlopen
import os
import pandas as pd

class ImageTextSimilarity_bio:
    def __init__(self):
        # Loading the BiomedCLIP model
        self.model, self.preprocess, _ = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        self.tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

        # Configure device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        print("biomedclip model loaded successfully.")

    def calculate_similarity(self, image_path, text_folder):
        # Prepare image
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)

        # Prepare texts and load file names
        texts, text_files = self.prepare_texts(text_folder)

        # Model inference
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(texts)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            scores = (100.0 * image_features @ text_features.T).softmax(dim=-1).flatten().tolist()

        # Combine image, text files and scores into a DataFrame
        results = pd.DataFrame({
            'Image': [image_path] * len(text_files),
            'Text File': [os.path.join(text_folder, file) for file in text_files],
            'Similarity Score': scores
        })

        return results

    def calculate_similarity_path(self, image_path, text):
        # Prepare image
        image = Image.open(urlopen(image_path) if image_path.startswith('http') else image_path)
        image = self.preprocess(image).unsqueeze(0).to(self.device)

        # Prepare text
        tokenized_text = self.tokenizer([f'this is a photo of {text}'], context_length=256).to(self.device)

        # Model inference
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(tokenized_text)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            scores = (100.0 * image_features @ text_features.T).softmax(dim=-1).flatten().tolist()

        # Result as a dictionary
        result = {
            'Image': image_path,
            'Text': text,
            'Similarity Score': scores[0]
        }

        return result

    def calculate_similarity_hf(self, image, text):
        # Check if the image is already a PIL.Image object
        if not isinstance(image, Image.Image):
            # If it's a path or URL, open it
            image = Image.open(urlopen(image) if image.startswith('http') else image)

        # Prepare image using model-specific preprocessing
        image = self.preprocess(image).unsqueeze(0).to(self.device)

        # Prepare text
        tokenized_text = self.tokenizer([f'this is a photo of {text}'], context_length=256).to(self.device)

        # Model inference
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(tokenized_text)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            scores = (100.0 * image_features @ text_features.T).softmax(dim=-1).flatten().tolist()

        # Result as a dictionary
        result = {
            'Image': 'In-memory image' if isinstance(image, Image.Image) else 'Image path',
            'Text': text,
            'Similarity Score': scores[0]
        }

        return result

    def prepare_texts(self, text_folder):
        # List of text files corresponding to different categories or descriptions
        text_files = [f for f in os.listdir(text_folder) if f.endswith('.txt')]
        labels = []
        for text_file in text_files:
            path = os.path.join(text_folder, text_file)
            with open(path, 'r') as file:
                labels.append(file.read().strip())

        # Tokenize texts
        texts = self.tokenizer([f'this is a photo of {label}' for label in labels], context_length=256)
        texts = texts.to(self.device)
        return texts, text_files
