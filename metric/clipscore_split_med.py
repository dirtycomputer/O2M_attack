import csv
import torch
from open_clip import (
    create_model_from_pretrained,
    get_tokenizer,
    create_model_and_transforms,
)
import datasets
import argparse
from typing import Any, List, Tuple
import time


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Function '{func.__name__}' executed in {end - start:.4f} seconds.")
        return result

    return wrapper


def clip_score(model, image, tokenized_text) -> float:
    with torch.no_grad():
        image_features, text_features, logit_scale = model(image, tokenized_text)
        score = (logit_scale * image_features @ text_features.t()).detach()
    return score.item()


def process_split(
    split: str, device: torch.device, model: Any, preprocess: Any, tokenizer: Any
) -> List[Tuple]:
    dataset = datasets.load_dataset("MedMLLM-attack/3MAD-70K")
    results = []

    for index, example in enumerate(dataset[split]):
        id, file_name, original_attribute, unmatch_attribute, image = (
            example["id"],
            example["file_name"],
            example["original_attribute"],
            example["unmatch_attribute"],
            example["image"],
        )
        tokenized_text_original = tokenizer(
            [f"this is a photo of {original_attribute}"]
        ).to(device)
        tokenized_text_unmatch = tokenizer(
            [f"this is a photo of {unmatch_attribute}"]
        ).to(device)
        image = torch.stack([preprocess(image)]).to(device)
        assert image.shape[-2:] == (224, 224)

        origin_score = clip_score(model, image, tokenized_text_original)
        unmatch_score = clip_score(model, image, tokenized_text_unmatch)

        results.append(
            (
                id,
                file_name,
                original_attribute,
                origin_score,
                unmatch_attribute,
                unmatch_score,
            )
        )
        print(f"Processed {split} split, index {index}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate BiomedCLIP model on MedMLLM-attack/3MAD-70K dataset."
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        help="Dataset split to process ('train', 'test', etc.)",
    )
    parser.add_argument(
        "--device",
        type=str,
        required=True,
        help="CUDA device ('cuda:0', 'cuda:1', etc.)",
    )
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}")
    model, _, preprocess = create_model_and_transforms(
        "ViT-B-16", pretrained="datacomp_xl_s13b_b90k"
    )
    tokenizer = get_tokenizer('ViT-B-16')

    # model_name = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    # tokenizer_name = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    # model, preprocess = create_model_from_pretrained(model_name)
    # tokenizer = get_tokenizer(tokenizer_name)
    model.to(device)
    model.eval()
    print("Model and tokenizer loaded successfully.")

    results = process_split(args.split, device, model, preprocess, tokenizer)
    print(results)
    # with open(f"metric/img2text_results/{args.split}.csv", mode="w", newline="") as file:
    #     writer = csv.writer(file)
    #     writer.writerow(
    #         [
    #             "id",
    #             "file_name",
    #             "original_attribute",
    #             "origin_score",
    #             "unmatch_attribute",
    #             "unmatch_score",
    #         ]
    #     )
    #     writer.writerows(results)
