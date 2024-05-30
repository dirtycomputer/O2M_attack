import pandas as pd
from text2text_similarity import TextSimilarityCalculator
from image2text_similarity import ImageTextSimilarity  
from BiomedCLIP import ImageTextSimilarity_bio
import os

# 指定文本文件夹路径
normal_text_folder = './output_normal'
harmful_text_folder = './output_harmful'
image_folder = './images'

# 创建相似度计算器实例
similarity_calculator_t2t = TextSimilarityCalculator()
similarity_calculator_i2t = ImageTextSimilarity()  
similarity_calculator_i2t_bio = ImageTextSimilarity_bio()

# 计算文本间的相似度并打印结果
results = similarity_calculator_t2t.compute_similarities_folder(normal_text_folder, harmful_text_folder)
print(results)

# 定义一个函数用来计算并输出每张图片与两种文本的相似度分数
def calculate_and_print_image_text_similarity(image_folder, normal_text_folder, harmful_text_folder):
    all_results = []  # 用于收集所有结果的列表

    for image_file in os.listdir(image_folder):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, image_file)
            
            # 计算每个模型的相似度
            results_normal = similarity_calculator_i2t.calculate_similarity(image_path, normal_text_folder)
            results_harmful = similarity_calculator_i2t.calculate_similarity(image_path, harmful_text_folder)
            results_normal_bio = similarity_calculator_i2t_bio.calculate_similarity(image_path, normal_text_folder)
            results_harmful_bio = similarity_calculator_i2t_bio.calculate_similarity(image_path, harmful_text_folder)
            
            # 整合相同图片和文本的结果
            for i in range(len(results_normal)):
                normal_row = results_normal.iloc[i]
                normal_bio_row = results_normal_bio.iloc[i]
                harmful_row = results_harmful.iloc[i]
                harmful_bio_row = results_harmful_bio.iloc[i]

                combined_row = {
                    'Image': image_path,
                    'Text File': normal_row['Text File'],
                    'Similarity Score (CLIP-ViT)': normal_row['Similarity Score'],
                    'Similarity Score (BiomedCLIP)': normal_bio_row['Similarity Score']
                }
                all_results.append(combined_row)

                combined_row_harmful = {
                    'Image': image_path,
                    'Text File': harmful_row['Text File'],
                    'Similarity Score (CLIP-ViT)': harmful_row['Similarity Score'],
                    'Similarity Score (BiomedCLIP)': harmful_bio_row['Similarity Score']
                }
                all_results.append(combined_row_harmful)

    # 创建一个DataFrame来显示所有结果
    results_df = pd.DataFrame(all_results)
    print(results_df)

# 执行函数
calculate_and_print_image_text_similarity(image_folder, normal_text_folder, harmful_text_folder)
