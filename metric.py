from FlagEmbedding import BGEM3FlagModel
from PIL import Image
import open_clip
import torch


# https://github.com/FlagOpen/FlagEmbedding
class TxtSimCal:
    def __init__(self, model_path, use_fp16=True, 
                 max_passage_length = 128, weights_for_different_modes = [0.4, 0.2, 0.4]):
        self.model = BGEM3FlagModel(model_path, use_fp16=use_fp16)
        self.max_passage_length = max_passage_length
        self.weights_for_different_modes = weights_for_different_modes
    
    def compute_score(self, question, response):
        assert type(question) == str and type(response) == str
        # 'colbert', 'sparse', 'dense', 'sparse+dense', 'colbert+sparse+dense'
        return self.model.compute_score([[question, response]], 
                                        max_passage_length = self.max_passage_length, 
                                        weights_for_different_modes = self.weights_for_different_modes)["colbert+sparse+dense"][0]
        
# https://github.com/mlfoundations/open_clip
class ImgSimCal:
    def __init__(self, model_path):
        assert model_path in ["ViT-B-16", "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"]
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_path)
        self.tokenizer = open_clip.get_tokenizer(model_path)
        self.model.eval()
        self.model.cuda()
    
    def compute_score(self, image_path, response):
        assert type(image_path) == str and type(response) == str and image_path.lower().endswith(('.jpg', '.png'))

        image = self.preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).cuda()
        text = self.tokenizer([response]).cuda()

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            score = (100.0 * image_features @ text_features.T)
        
        return score.item()
    
    
if __name__ == '__main__':
    image_path = "test.jpg"  # 替换为你的测试图像路径
    question = "A description of the image."
    response = "A description of the image."
    model_path = "BAAI/bge-m3"
    txt_sim_cal = TxtSimCal(model_path) 
    score = txt_sim_cal.compute_score(question, response)
    
    # model_path = "ViT-B-16"  # 选择合适的模型
    # model_path = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    # img_sim_cal = ImgSimCal(model_path)


    # score = img_sim_cal.compute_score(image_path, response)
    print(f"Similarity score: {score}")