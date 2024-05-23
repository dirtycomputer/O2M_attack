from pathlib import Path
import torch
from torchvision import transforms
import wandb
from llava.gcg.opti_utils import test_prefixes, check_for_attack_success, get_filtered_cands, get_logits, sample_control, target_loss, token_gradients


def gcg_attack(num_steps, model, tokenizer, suffix_manager, adv_suffix, images, test_prefixes, batch_size=10, topk=64):
    for i in range(num_steps):

        input_ids = suffix_manager.get_input_ids(adv_string=adv_suffix)
        input_ids = input_ids.to(model.device)

        coordinate_grad = token_gradients(
            model,
            input_ids,
            suffix_manager._control_slice,
            suffix_manager._target_slice,
            suffix_manager._loss_slice,
            images
        )

        with torch.no_grad():

            adv_suffix_tokens = input_ids[suffix_manager._control_slice].to(model.device) 

            new_adv_suffix_toks = sample_control(
                adv_suffix_tokens,
                coordinate_grad,
                batch_size,
                topk=topk,
                temp=1,
                not_allowed_tokens=None,
            )

            new_adv_suffix = get_filtered_cands(
                tokenizer, new_adv_suffix_toks, filter_cand=True, curr_control=adv_suffix
            )
            
            logits, ids = get_logits(
                model=model,
                tokenizer=tokenizer,
                input_ids=input_ids,
                images=images,
                control_slice=suffix_manager._control_slice,
                test_controls=new_adv_suffix,
                return_ids=True,
            )

            losses = target_loss(logits, ids, suffix_manager._target_slice)

            best_new_adv_suffix_id = losses.argmin()
            best_new_adv_suffix = new_adv_suffix[best_new_adv_suffix_id]

            current_loss = losses[best_new_adv_suffix_id].item()

            adv_suffix = best_new_adv_suffix
            is_success, gen_str = check_for_attack_success(
                model,
                tokenizer,
                suffix_manager.get_input_ids(adv_string=adv_suffix).to(model.device),
                suffix_manager._assistant_role_slice,
                test_prefixes,
                images
            )
        
    return is_success, current_loss, gen_str, adv_suffix, 


def pgd_attack(num_steps, model, tokenizer, suffix_manager, adv_suffix, images, test_prefixes, eps=0.3, alpha=2/255) :
    
    input_ids = suffix_manager.get_input_ids(adv_string=adv_suffix).unsqueeze(0).to(model.device)
    images = images.to(model.device)
    ori_images = images.clone().detach().to(model.device)
        
    for i in range(num_steps) :
        if images.grad is not None:
            images.grad.zero_()
        images.requires_grad_().retain_grad()
        outputs = model(
            input_ids = input_ids,
            images = images,
        )
        losses = target_loss(outputs.logits, input_ids, suffix_manager._target_slice)
        losses.backward()

        adv_images = images - alpha * images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
        
        is_success, gen_str = check_for_attack_success(
                model,
                tokenizer,
                suffix_manager.get_input_ids(adv_string=adv_suffix).to(model.device),
                suffix_manager._assistant_role_slice,
                test_prefixes,
                images
            )
        current_loss = losses.item()
    noise = images - ori_images
    return is_success, current_loss, gen_str, noise



def save_results(sample_attr, adv_suffix, images, gen_str, mode):
    root = f'results/{mode}/{sample_attr[-2]}'
    Path(root).mkdir(parents=True, exist_ok=True)
    torch.save(images, f'{root}/{sample_attr[0]}.pt')
    norm_images = ((images - images.min()) / (images.max() - images.min())).squeeze()
    transforms.ToPILImage()(norm_images).save(f'{root}/{sample_attr[0]}.jpg')
    with open(f'{root}/question_{sample_attr[0]}.txt', 'w') as file:
        if mode=="pgd":
            file.write(sample_attr[-1])
        else:
            file.write(sample_attr[-1]+adv_suffix)
    with open(f'{root}/gen_{sample_attr[0]}.txt', 'w') as file:
        file.write(gen_str)

def run_mcm_attack(sample_attr, table, num_steps, model, tokenizer, suffix_manager, adv_suffix, images, test_prefixes):
    noise = torch.zeros_like(images)
    mode = None
    
    for i in range(num_steps):
        gcg_is_success, gcg_loss, gcg_gen_str, gcg_adv_str = gcg_attack(
            num_steps=1, model=model, tokenizer=tokenizer, suffix_manager=suffix_manager,
            adv_suffix=adv_suffix, test_prefixes=test_prefixes, images=images,
        )
        pgd_is_success, pgd_loss, pgd_gen_str, pgd_noise = pgd_attack(
            num_steps=5, model=model, tokenizer=tokenizer, suffix_manager=suffix_manager,
            adv_suffix=adv_suffix, images=images, test_prefixes=test_prefixes
        )
        
        if gcg_loss < pgd_loss:
            is_success, current_loss, gen_str, adv_str = gcg_is_success, gcg_loss, gcg_gen_str, gcg_adv_str
            adv_suffix = adv_str
            mode = "gcg"
            print("gcg attack")
        else:
            is_success, current_loss, gen_str, noise = pgd_is_success, pgd_loss, pgd_gen_str, pgd_noise
            images = images + pgd_noise
            noise = pgd_noise
            mode = "pgd"
            print("pgd attack")
        print("\033[1;31m" + "is_success:" + "\033[0m", is_success)
        print("\033[1;32m" + "current_loss:" + "\033[0m", current_loss)
        print("\033[1;33m" + "gen_str:" + "\033[0m", gen_str)
        print("\033[1;34m" + "adv_str:" + "\033[0m", adv_suffix)
        print("\033[1;35m" + "noise_mean:" + "\033[0m", noise.mean())
        table.add_data(*sample_attr, adv_suffix, wandb.Image(noise), wandb.Image(images), gen_str, int(is_success), current_loss, mode, i)
    save_results(sample_attr, adv_suffix, images, gen_str, "mcm")
    return table




def run_pgd_attack(sample_attr, table, num_steps, model, tokenizer, suffix_manager, adv_suffix, images, test_prefixes):
    noise = torch.zeros_like(images)
    mode = None
    
    for i in range(num_steps):
        pgd_is_success, pgd_loss, pgd_gen_str, pgd_noise = pgd_attack(
            num_steps=1, model=model, tokenizer=tokenizer, suffix_manager=suffix_manager,
            adv_suffix=adv_suffix, images=images, test_prefixes=test_prefixes
        )
        

        is_success, current_loss, gen_str, noise = pgd_is_success, pgd_loss, pgd_gen_str, pgd_noise
        images = images + pgd_noise
        noise = pgd_noise
        mode = "pgd"
        print("pgd attack")
        
        print("\033[1;31m" + "is_success:" + "\033[0m", is_success)
        print("\033[1;32m" + "current_loss:" + "\033[0m", current_loss)
        print("\033[1;33m" + "gen_str:" + "\033[0m", gen_str)
        print("\033[1;34m" + "adv_str:" + "\033[0m", adv_suffix)
        print("\033[1;35m" + "noise_mean:" + "\033[0m", noise.mean())
        table.add_data(*sample_attr, adv_suffix, wandb.Image(noise), wandb.Image(images), gen_str, int(is_success), current_loss, mode, i)
    save_results(sample_attr, adv_suffix, images, gen_str, "pgd")
    return table



def run_gcg_attack(sample_attr, table, num_steps, model, tokenizer, suffix_manager, adv_suffix, images, test_prefixes):
    noise = torch.zeros_like(images)
    mode = None
    
    for i in range(num_steps):
        gcg_is_success, gcg_loss, gcg_gen_str, gcg_adv_str = gcg_attack(
            num_steps=1, model=model, tokenizer=tokenizer, suffix_manager=suffix_manager,
            adv_suffix=adv_suffix, test_prefixes=test_prefixes, images=images,
        )
        
        is_success, current_loss, gen_str, adv_str = gcg_is_success, gcg_loss, gcg_gen_str, gcg_adv_str
        adv_suffix = adv_str
        mode = "gcg"
        print("gcg attack")

        print("\033[1;31m" + "is_success:" + "\033[0m", is_success)
        print("\033[1;32m" + "current_loss:" + "\033[0m", current_loss)
        print("\033[1;33m" + "gen_str:" + "\033[0m", gen_str)
        print("\033[1;34m" + "adv_str:" + "\033[0m", adv_suffix)
        print("\033[1;35m" + "noise_mean:" + "\033[0m", noise.mean())
        table.add_data(*sample_attr, adv_suffix, wandb.Image(noise), wandb.Image(images), gen_str, int(is_success), current_loss, mode, i)
    save_results(sample_attr, adv_suffix, images, gen_str, "gcg")
    return table

