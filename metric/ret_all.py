import pandas as pd
import torch
import open_clip
from open_clip import create_model_and_transforms, get_tokenizer, create_model_from_pretrained
from PIL import Image
import os
from tqdm import tqdm

# 确保正确导入你的TextSimilarityCalculator类
from text2text_similarity import TextSimilarityCalculator  

folder_paths = [
    '../llavamed_ret_patch'
    # '../ret_pgd'
]

def clip_score_image(model, preprocess, tokenizer, device, image_path, text):
    image = Image.open(image_path).convert('RGB')
    image = preprocess(image).unsqueeze(0).to(device)
    tokenized_text = tokenizer([text]).to(device)
    
    with torch.no_grad():
        image_features, text_features, logit_scale = model(image, tokenized_text)
        score = (logit_scale * image_features @ text_features.t()).detach()
    return score.item()

def process_csv(input_csv, output_csv, model_bio, preprocess_bio, tokenizer_bio, device, text_similarity_calculator):
    df = pd.read_csv(input_csv)
    results = []

    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing rows"):
        id = row['id']
        file_name = row['file_name']
        original_attribute = str(row['original_attribute'])
        unmatch_attribute = str(row['unmatch_attribute'])
        normal_prompt = str(row['normal_prompt'])
        harmful_prompt = str(row['harmful_prompt'])
        policy = str(row['policy'])
        gen_str = str(row['gen_str'])
        current_loss = str(row['current_loss'])
        mode = str(row['mode'])
        iter = str(row['iter'])
        malicious_or_unmatch = str(row['malicious_or_unmatch'])

        image_path = f"/home/huangxijie/MedMLLM_attack/{file_name}"

        text_unmatch_score = None
        text_malicious_score = None
        text_both_score = None
        img_score_unmatch = None
        img_score_malicious = None
        img_score_both = None

        if malicious_or_unmatch == 'malicious':
            text_malicious_pair = f"{original_attribute} {harmful_prompt}"
            text_malicious_score = text_similarity_calculator.compute_similarity_text(gen_str, text_malicious_pair)
            img_score_malicious = clip_score_image(model_bio, preprocess_bio, tokenizer_bio, device, image_path, gen_str)
        elif malicious_or_unmatch == 'unmatch':
            text_unmatch_pair = f"{unmatch_attribute} {normal_prompt}"
            text_unmatch_score = text_similarity_calculator.compute_similarity_text(gen_str, text_unmatch_pair)
            img_score_unmatch = clip_score_image(model_bio, preprocess_bio, tokenizer_bio, device, image_path, gen_str)
        elif malicious_or_unmatch == 'both':
            text_both_pair = f"{unmatch_attribute} {harmful_prompt}"
            text_both_score = text_similarity_calculator.compute_similarity_text(gen_str, text_both_pair)
            img_score_both = clip_score_image(model_bio, preprocess_bio, tokenizer_bio, device, image_path, gen_str)

        results.append([
            id, file_name, current_loss, iter, mode, malicious_or_unmatch, 
            img_score_malicious, img_score_unmatch, img_score_both,
            text_unmatch_score.iloc[0, 4] if text_unmatch_score is not None else '',
            text_malicious_score.iloc[0, 4] if text_malicious_score is not None else '',
            text_both_score.iloc[0, 4] if text_both_score is not None else ''
        ])

    result_df = pd.DataFrame(results, columns=[
        "id", "filename", "current_loss", "iter", "mode", "malicious_or_unmatch", 
        "img_score_malicious", "img_score_unmatch","img_score_both", "text_unmatch_score", "text_malicious_score","text_both_score"
    ])
    result_df.to_csv(output_csv, index=False)
    print(f"CSV文件已成功生成在: {output_csv}")

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model_nature, _, preprocess_nature = open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai')
    # tokenizer_nature = open_clip.get_tokenizer('ViT-B-16')
    # model_nature.to(device)
    # model_nature.eval()
    
    model_name = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    tokenizer_name = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    model_bio, preprocess_bio = create_model_from_pretrained(model_name)
    tokenizer_bio = get_tokenizer(tokenizer_name)
    model_bio.to(device)
    model_bio.eval()

    text_similarity_calculator = TextSimilarityCalculator()

    for folder_path in folder_paths:
        for filename in os.listdir(folder_path):
            if filename.endswith(".csv"):
                input_csv_path = os.path.join(folder_path, filename)
                output_csv_path = os.path.join(folder_path, f"processed_{filename}")
                process_csv(input_csv_path, output_csv_path, model_bio, preprocess_bio, tokenizer_bio, device, text_similarity_calculator)
    # input_csv_path = "../llavamed_ret_patch/a.csv"
    # output_csv_path = "../llavamed_ret_patch/1.csv"
    # process_csv(input_csv_path, output_csv_path, model_bio, preprocess_bio, tokenizer_bio, device, text_similarity_calculator)