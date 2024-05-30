import pandas as pd
import torch
import open_clip
from open_clip import create_model_and_transforms, get_tokenizer,create_model_from_pretrained
from PIL import Image
from text2text_similarity import TextSimilarityCalculator  # 确保正确导入你的TextSimilarityCalculator类
import os
from tqdm import tqdm

def clip_score_image(model, preprocess, tokenizer, device, image_path, text):
    image = Image.open(image_path).convert('RGB')
    image = preprocess(image).unsqueeze(0).to(device)  # Apply preprocessing and add batch dimension
    # print(image.shape)
    
    # Tokenize text and move it to the correct device
    tokenized_text = tokenizer([text])
    tokenized_text = tokenized_text.to(device)  # Move tensor to the device
    # print(tokenized_text.shape)
    with torch.no_grad():
        image_features, text_features, logit_scale = model(image, tokenized_text)
        # print(f"Logit scale: {logit_scale.item()}")
        score = (logit_scale* image_features @ text_features.t()).detach()
        # print(score.item())
    return score.item()


def process_csv(input_csv, output_csv, model_nature, preprocess_nature,tokenizer_nature,model_bio, preprocess_bio,tokenizer_bio, device, text_similarity_calculator):
    df = pd.read_csv(input_csv)
    results = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing rows"):
        
        mal_resp = str(row['malicious_response'])[:128]
        harm_prompt = f"{str(row['original_attribute'])[:128]} {str(row['harmful_prompt'])[:128]}"
        unmat_resp = str(row['unmatch_response'])[:128]
        norm_prompt = f"{str(row['unmatch_attribute'])[:128]} {str(row['normal_prompt'])[:128]}"
        image_path = f"/home/huangxijie/MedMLLM_attack/{str(row['file_name'])[:128]}"   # Assuming this is a full path to the image

        # 使用TextSimilarityCalculator计算文本相似度
        text_sim_score1 = text_similarity_calculator.compute_similarity_text(mal_resp, harm_prompt)
        text_sim_score2 = text_similarity_calculator.compute_similarity_text(unmat_resp, norm_prompt)

        # 计算图片和文本之间的相似度
        image_text_sim_score1 = clip_score_image(model_nature, preprocess_nature,tokenizer_nature, device, image_path, mal_resp)
        image_text_sim_score2 = clip_score_image(model_nature, preprocess_nature,tokenizer_nature, device, image_path, unmat_resp)
        image_text_sim_score3 = clip_score_image(model_bio, preprocess_bio,tokenizer_bio, device, image_path, mal_resp)
        image_text_sim_score4 = clip_score_image(model_bio, preprocess_bio,tokenizer_bio, device, image_path, unmat_resp)
        results.append([
            row['id'], text_sim_score1.iloc[0,4].tolist(), text_sim_score2.iloc[0,4].tolist(), image_text_sim_score1, image_text_sim_score2,image_text_sim_score3,image_text_sim_score4
        ])
        # print(results[index])
    result_df = pd.DataFrame(results, columns=['id', 'text_sim_score_malicious', 'text_sim_score_unmatch', 'image_text_sim_score_malicious_nature', 'image_text_sim_score_unmatch_nature','image_text_sim_score_malicious_bio', 'image_text_sim_score_unmatch_nature_bio'])
    result_df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_nature, _, preprocess_nature = open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai')
    tokenizer_nature = open_clip.get_tokenizer('ViT-B-16')
    model_nature.to(device)
    model_nature.eval()
    print("Model and tokenizer loaded successfully.")
    
    model_name = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    tokenizer_name = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    model_bio, preprocess_bio = create_model_from_pretrained(model_name)
    tokenizer_bio = get_tokenizer(tokenizer_name)
    model_bio.to(device)
    model_bio.eval()
    print("Model and tokenizer loaded successfully.")

    # 实例化文本相似度计算器
    text_similarity_calculator = TextSimilarityCalculator()


    directory_path = 'normal_ret'
    for dirpath, dirnames, filenames in os.walk(directory_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            print(f"Processing file: {file_path}")
            input_csv_path = file_path
            output_csv_path = f"./ret_com/ret{os.path.basename(file_path)}"
            process_csv(input_csv_path, output_csv_path, model_nature, preprocess_nature,tokenizer_nature,model_bio, preprocess_bio,tokenizer_bio, device, text_similarity_calculator)
