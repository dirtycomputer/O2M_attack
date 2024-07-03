import argparse
import pandas as pd
from PIL import Image
import csv

import wandb

from llava.attacks.attacks import run_gcg_attack, run_mcm_attack, run_pgd_attack
from llava.gcg.conversation import conversation_multimodal
from llava.gcg.opti_utils import test_prefixes
from llava.gcg.string_utils import SuffixManager
from llava.model.llava import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN, load_model_tokenizer

from datasets import load_dataset


def eval_model(args):
    attack_mode = args.attack_mode
    wandb.login(key="replace your wandb key")
    wandb.init(project=f"unmatch_malicious_{attack_mode}")

    device = f"cuda:{args.device}"
    dataset = load_dataset(args.dataset, num_proc=18)
    tokenizer, model, image_processor = load_model_tokenizer(args.model_name, device)
    conv = conversation_multimodal.copy()
    
    split = args.split
    split_data = dataset[split]

    adv_suffix = args.adv_suffix
    
    # malicious_table = wandb.Table(columns=[
    #     "id", "file_name", "image", "original_attribute", "unmatch_attribute", "normal_prompt", "harmful_prompt", "policy", "keyphrases",
    #     "target", "malicious_or_unmatch", "question",
    #     "adv_suffix", "noise", "adv_image", "gen_str", "is_success", "current_loss", "mode", "iter"]
    # )
    
    # unmatch_table = wandb.Table(columns=[
    #     "id", "file_name", "image", "original_attribute", "unmatch_attribute", "normal_prompt", "harmful_prompt", "policy", "keyphrases",
    #     "target", "malicious_or_unmatch", "question",
    #     "adv_suffix", "noise", "adv_image", "gen_str", "is_success", "current_loss", "mode", "iter"]
    # )
    
    both_table = wandb.Table(columns=[
        "id", "file_name", "image", "original_attribute", "unmatch_attribute", "normal_prompt", "harmful_prompt", "policy", "keyphrases",
        "target", "malicious_or_unmatch", "question",
        "adv_suffix", "noise", "adv_image", "gen_str", "is_success", "current_loss", "mode", "iter"]
    )
    
    for index, sample in enumerate(split_data):
        
        print(index)
        id = sample["id"]
        file_name = sample["file_name"]
        image = sample["image"]
        image.convert('RGB').save(f'results/images/{id}.jpg', 'JPEG')
        original_attribute = sample["original_attribute"]
        unmatch_attribute = sample["unmatch_attribute"]
        normal_prompt = sample["normal_prompt"]
        harmful_prompt = sample["harmful_prompt"]
        policy = sample["policy"]
        keyphrases = sample["key_phrases"]
        target = "Sure, here is how to " + (keyphrases if keyphrases is not None else "")
        images = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0].unsqueeze(0).half().to(device)

        # # malicious
        # malicious_or_unmatch = "malicious"
        # malicious_question = "This image is about " + original_attribute+ ". "+ harmful_prompt+"\n"
        # malicious_instruction = malicious_question + DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * image_processor.image_token_len + DEFAULT_IM_END_TOKEN + "\n"
        # malicious_policy = policy
        # malicious_suffix_manager = SuffixManager(tokenizer=tokenizer, conv_template=conv, instruction=malicious_instruction, adv_string=adv_suffix, target=target)
        # malicious_sample_attr = id, file_name, wandb.Image(image), original_attribute, unmatch_attribute, normal_prompt, harmful_prompt, malicious_policy, keyphrases, target, malicious_or_unmatch, malicious_question
        

        # malicious_attack_params = {
        #     "sample_attr": malicious_sample_attr,
        #     "table": malicious_table,
        #     "num_steps": args.iter,
        #     "model": model,
        #     "tokenizer": tokenizer,
        #     "suffix_manager": malicious_suffix_manager,
        #     "adv_suffix": adv_suffix,
        #     "images": images,
        #     "test_prefixes": test_prefixes
        # }
        
        # if attack_mode == "mcm":
        #     malicious_table = run_mcm_attack(**malicious_attack_params)
        # elif attack_mode == "pgd":
        #     malicious_table = run_pgd_attack(**malicious_attack_params)
        # elif attack_mode == "gcg":
        #     malicious_table = run_gcg_attack(**malicious_attack_params)
        
        
        # # unmatch
        # malicious_or_unmatch = "unmatch"
        # unmatch_question = "This image is about " + unmatch_attribute + ". "+ normal_prompt+"\n"
        # unmatch_instruction = unmatch_question + DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * image_processor.image_token_len + DEFAULT_IM_END_TOKEN + "\n"
        # unmatch_policy = "Unmatch"
        # unmatch_suffix_manager = SuffixManager(tokenizer=tokenizer, conv_template=conv, instruction=unmatch_instruction, adv_string=adv_suffix, target=target)
        # unmatch_sample_attr = id, file_name, wandb.Image(image), original_attribute, unmatch_attribute, normal_prompt, harmful_prompt, unmatch_policy, keyphrases, target, malicious_or_unmatch, unmatch_question
        

        # unmatch_attack_params = {
        #     "sample_attr": unmatch_sample_attr,
        #     "table": unmatch_table,
        #     "num_steps": args.iter,
        #     "model": model,
        #     "tokenizer": tokenizer,
        #     "suffix_manager": unmatch_suffix_manager,
        #     "adv_suffix": adv_suffix,
        #     "images": images,
        #     "test_prefixes": test_prefixes
        # }
        
        # if attack_mode == "mcm":
        #     unmatch_table = run_mcm_attack(**unmatch_attack_params)
        # elif attack_mode == "pgd":
        #     unmatch_table = run_pgd_attack(**unmatch_attack_params)
        # elif attack_mode == "gcg":
        #     unmatch_table = run_gcg_attack(**unmatch_attack_params)
            
            
        # both
        malicious_or_unmatch = "both"
        both_question = "This image is about " + unmatch_attribute + ". "+ harmful_prompt+"\n"
        both_instruction = both_question + DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * image_processor.image_token_len + DEFAULT_IM_END_TOKEN + "\n"
        both_policy = "Both"
        both_suffix_manager = SuffixManager(tokenizer=tokenizer, conv_template=conv, instruction=both_instruction, adv_string=adv_suffix, target=target)
        both_sample_attr = id, file_name, wandb.Image(image), original_attribute, unmatch_attribute, normal_prompt, harmful_prompt, both_policy, keyphrases, target, malicious_or_unmatch, both_question
        

        both_attack_params = {
            "sample_attr": both_sample_attr,
            "table": both_table,
            "num_steps": args.iter,
            "model": model,
            "tokenizer": tokenizer,
            "suffix_manager": both_suffix_manager,
            "adv_suffix": adv_suffix,
            "images": images,
            "test_prefixes": test_prefixes
        }
        
        if attack_mode == "mcm":
            both_table = run_mcm_attack(**both_attack_params)
        elif attack_mode == "pgd":
            both_table = run_pgd_attack(**both_attack_params)
        elif attack_mode == "gcg":
            both_table = run_gcg_attack(**both_attack_params)
            
        
    # wandb.log({f"{split}_malicious": malicious_table})
    # wandb.log({f"{split}_unmatch": unmatch_table})
    wandb.log({f"{split}_both": both_table})
    wandb.run.summary["split"] = split
    wandb.run.summary["attack_mode"] = attack_mode
    
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="/data/model/LLaVA_Med_weight/")
    parser.add_argument("--dataset", type=str, default="/data/dataset/3MAD-Tiny-1K")
    parser.add_argument("--iter", type=int, default=10)
    parser.add_argument("--device", type=str, default="9")
    parser.add_argument("--adv-suffix", type=str, default="}&"*10)
    parser.add_argument("--split", type=str, default="CT_Chest")
    parser.add_argument("--attack-mode", type=str, default="mcm")
    args = parser.parse_args()

    eval_model(args)
