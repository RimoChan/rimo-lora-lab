import os
import time
import json
import pickle
import logging
import itertools
from pathlib import Path

from typing import Any

import fire
import requests
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import ImageDraw, ImageFont
from tqdm.auto import tqdm

from diffusers import DDPMScheduler, StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from diffusers.loaders import StableDiffusionLoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params, compute_snr
from diffusers.utils import convert_state_dict_to_diffusers, convert_unet_state_dict_to_peft
from diffusers.utils.torch_utils import is_compiled_module

from compel import Compel, ReturnedEmbeddingsType
from common import clean, 生成optimizer, 哈, 哈哈, encode_prompt, compute_time_ids, 检查模型类型, 计时, buffered_iterator, add_image_jpeg, cosine_with_restart_scheduler改
from data import 读取数据集


def validation(global_step, accelerator, vae, text_encoder, text_encoder_2, unet, torch_dtype, trackers, pretrained_model_name_or_path, validation_prompt_list, current_model_index):
    if accelerator.is_main_process:
        with torch.inference_mode():
            for _ in range(1000):
                try:
                    pipeline = StableDiffusionXLPipeline.from_single_file(
                        pretrained_model_name_or_path,
                        vae=vae,
                        text_encoder=text_encoder,
                        text_encoder_2=text_encoder_2,
                        unet=unet,
                        torch_dtype=torch_dtype,
                    ).to(accelerator.device)
                    break
                except requests.exceptions.RequestException as e:
                    print('网络不好，休息1下！', repr(e))
                    time.sleep(30)
            pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
            pipeline.set_progress_bar_config(disable=True)
            clean()
            images = [
                pipeline(
                    prompt=prompt,
                    generator=torch.Generator(device=accelerator.device).manual_seed(100+i),
                    num_inference_steps=20,
                    guidance_scale=7,
                    width=768,
                    height=1024,
                ).images[0] for i, prompt in enumerate(validation_prompt_list)
            ]
            clean()
            ImageDraw.Draw(images[0]).text((10, 10), str(current_model_index), fill="red", font=ImageFont.load_default(128))
            for tracker in trackers:
                if tracker.name == "tensorboard":
                    add_image_jpeg(tracker.writer, "validation", np.concatenate([np.asarray(img) for img in images], axis=1), global_step)


def 加载模型(path) -> tuple:
    模型类型 = 检查模型类型(path)
    assert 模型类型 == 'sdxl'
    临时pipeline = StableDiffusionXLPipeline.from_single_file(path)
    text_encoder_one = 临时pipeline.text_encoder
    text_encoder_two = 临时pipeline.text_encoder_2
    compel = Compel(truncate_long_prompts=False, tokenizer=[临时pipeline.tokenizer, 临时pipeline.tokenizer_2], text_encoder=[text_encoder_one, text_encoder_two],  returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, requires_pooled=[False, True])
    for m in 临时pipeline.vae, 临时pipeline.unet, text_encoder_one, text_encoder_two:
        m.requires_grad_(False)
    noise_scheduler = DDPMScheduler.from_config(临时pipeline.scheduler.config)
    return text_encoder_one, text_encoder_two, compel, 临时pipeline.unet, 临时pipeline.vae, noise_scheduler


def 相位转移(unet, state_dict):
    udk = [*unet.state_dict().keys()]
    dd = {}
    for k, v in state_dict.items():
        if k in udk:
            dd[k] = v
        else:
            dd[k.replace(".bias", ".base_layer.bias").replace(".weight", ".base_layer.weight")] = v
    for k in dd:
        assert k in udk
    unet.load_state_dict(dd, strict=False)


def conditional_loss( model_pred: torch.Tensor, target: torch.Tensor, reduction: str = "mean", loss_type: str = "l2", huber_c: float = 0.1,):
    if loss_type == "l2":
        loss = F.mse_loss(model_pred, target, reduction=reduction)
    elif loss_type == "huber" or loss_type == "huber_scheduled":
        loss = huber_c * (torch.sqrt((model_pred - target) ** 2 + huber_c**2) - huber_c)
        if reduction == "mean":
            loss = torch.mean(loss)
        elif reduction == "sum":
            loss = torch.sum(loss)
    else:
        raise NotImplementedError(f"Unsupported Loss Type {loss_type}")
    return loss


def vae_encode_with_cache(vae, pixel_values, cache_dir, 哈希值) -> torch.Tensor:
    p = Path(cache_dir) / f'{哈希值}.pkl'
    if p.exists():
        return pickle.load(open(p, 'rb'))
    res = vae.encode(pixel_values.to('cuda')).latent_dist.sample() * vae.config.scaling_factor
    with open(p, 'wb') as f:
        pickle.dump(res, f)
    return res


def main(
    pretrained_model_name_or_path: str | list[str],
    train_data_dir: str | list[str],
    validation_prompt_list: str | list[str] = '1girl',
    prior_loss_train_data_dir: str | list[str] = None,
    cache_dir: str = './rimo_lora_lab_cache',
    validation_steps: int = 100,
    prior_loss_steps: int = 8,
    output_dir: str = "lora",
    seed: int = None,
    batch_size: int = 1,
    max_train_steps: int = 10000,
    cycle_steps: int = 3000,
    checkpointing_steps: int = 500,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = False,
    lr: float = 1e-4,
    lr_scheduler: str = "constant_with_warmup",
    lr_warmup_steps: int = 50,
    lr_cosine_min: float = 0.1,
    snr_gamma: float = None,
    optimizer: str = "adam",
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 0.1,
    adam_epsilon: float = 1e-8,
    max_grad_norm: float = 1.0,
    prediction_type: str = None,
    mixed_precision: str = None,
    loss_type: str = "l2",
    huber_c: float = 0.1,
    rank: int = 32,
    alpha: int = None,
    time_min: int = 0,
    time_max: int = 1000,
    size_min: int = 704,
    size_max: int = 1280,
    drop_tag_rate: float = 0.0,
    drop_text_rate: float = 0.0,
    prompt_post_process: str = '',
    prior_loss_prompt_post_process: str = '',
    swap_every_n_steps: int = 8,
    resume_from_checkpoint: str = 'latest',
    use_mask: bool = False,
    mask_min: float = 0.1,
):
    alpha = alpha or rank // 2
    if seed is None:
        seed = int(time.time()) % 100
    metadata = {k: v for k, v in locals().items() if isinstance(v, (float, int, str)) and not k.startswith('_')}
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with='tensorboard',
        project_config=ProjectConfiguration(project_dir=output_dir, logging_dir=os.path.join(output_dir, 'logs')),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    set_seed(seed)
    if isinstance(pretrained_model_name_or_path, str):
        pretrained_model_name_or_path = pretrained_model_name_or_path.split(';')
    if isinstance(validation_prompt_list, str):
        validation_prompt_list = validation_prompt_list.split(';')
    if isinstance(train_data_dir, str):
        train_data_dir = train_data_dir.split(';')
    if isinstance(prior_loss_train_data_dir, str):
        prior_loss_train_data_dir = prior_loss_train_data_dir.split(';')
    if accelerator.is_main_process:
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    text_encoder_one, text_encoder_two, compel, unet, vae, noise_scheduler = 加载模型(pretrained_model_name_or_path[0])
    if len(pretrained_model_name_or_path) > 1:
        unet_state_dicts = [unet.to('cpu', dtype=weight_dtype).state_dict()]
        for p in pretrained_model_name_or_path[1:]:
            other_unet = 加载模型(p)[3]
            unet_state_dicts.append(other_unet.to('cpu', dtype=weight_dtype).state_dict())
            del other_unet

    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=torch.float32)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    if text_encoder_two:
        text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    unet.add_adapter(LoraConfig(
        r=rank,
        lora_alpha=alpha,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    ))

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            unet_lora_layers_to_save = None

            for model in models:
                if isinstance(unwrap_model(model), type(unwrap_model(unet))):
                    unet_lora_layers_to_save = convert_state_dict_to_diffusers(get_peft_model_state_dict(model))
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")
                if weights:
                    weights.pop()

            StableDiffusionXLPipeline.save_lora_weights(
                output_dir,
                unet_lora_layers=unet_lora_layers_to_save,
            )

    def load_model_hook(models, input_dir):
        unet_ = None

        while len(models) > 0:
            model = models.pop()
            if isinstance(unwrap_model(model), type(unwrap_model(unet))):
                unet_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict, _ = StableDiffusionLoraLoaderMixin.lora_state_dict(input_dir)
        unet_state_dict = {f"{k.replace('unet.', '')}": v for k, v in lora_state_dict.items() if k.startswith("unet.")}
        unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
        incompatible_keys = set_peft_model_state_dict(unet_, unet_state_dict, adapter_name="default")
        assert not incompatible_keys or not incompatible_keys.unexpected_keys, f'不对，{incompatible_keys.unexpected_keys}'
        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if mixed_precision == "fp16":
            models = [unet_]
            cast_training_params(models, dtype=torch.float32)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if mixed_precision == "fp16":
        models = [unet]
        cast_training_params(models, dtype=torch.float32)

    特 = [哈哈(train_data_dir), 哈哈(pretrained_model_name_or_path), f'{哈哈(prior_loss_train_data_dir)}_p{prior_loss_steps}' if prior_loss_train_data_dir else '', optimizer, f'snr{snr_gamma}', f'lr{lr}', f'drop{drop_tag_rate}_{drop_text_rate}' * (drop_tag_rate>0 or drop_text_rate>0), mixed_precision, lr_scheduler, f'{lr_cosine_min}' * (lr_scheduler == 'cosine_with_restarts'), f'lora{rank}_{alpha}', f'time{time_min}_{time_max}', f'size{size_min}_{size_max}', f'decay{adam_weight_decay}', 哈(prompt_post_process), 哈(prior_loss_prompt_post_process), f'mask{mask_min}' * use_mask, seed]
    特征 = '-'.join([str(i) for i in 特 if i != ''])

    optimizer = 生成optimizer(optimizer, unet, adam_beta1, adam_beta2, adam_weight_decay, adam_epsilon, lr, lr * 20)

    d = dict(
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps * 1000,
        num_cycles=round(max_train_steps * gradient_accumulation_steps * 1000 / cycle_steps),
    )
    if lr_scheduler == 'cosine_with_restarts':
        lr_scheduler = cosine_with_restart_scheduler改(**d, cosine_min=lr_cosine_min)
    else:
        lr_scheduler = get_scheduler(lr_scheduler, **d)

    unet, optimizer, lr_scheduler = accelerator.prepare(
        unet, optimizer, lr_scheduler
    )

    if accelerator.is_main_process:
        accelerator.init_trackers(特征)
    checkpoint_dir = os.path.join(output_dir, 特征)
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(Path(checkpoint_dir) / 'metadata.json', 'w', encoding='utf8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print('开始训练: ', str((Path(checkpoint_dir) / 'metadata.json').resolve()))
    global_step = 0
    if resume_from_checkpoint == 'latest':
        if 候选checkpoint := [*Path(checkpoint_dir).glob('checkpoint-*')]:
            resume_from_checkpoint = str(max(候选checkpoint, key=lambda i: int(i.stem.split('-')[-1])))
        else:
            resume_from_checkpoint = None
    if resume_from_checkpoint:
        accelerator.load_state(resume_from_checkpoint)
        global_step = int(resume_from_checkpoint.split("-")[-1])

    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    源 = buffered_iterator(itertools.chain.from_iterable(zip(*[
        读取数据集(i, drop_tag_rate=drop_tag_rate, drop_text_rate=drop_text_rate, size_min=size_min, size_max=size_max, prompt_post_process=prompt_post_process, use_mask=use_mask)
        for i in train_data_dir
    ])))
    if prior_loss_train_data_dir:
        源p = buffered_iterator(itertools.chain.from_iterable(zip(*[
            读取数据集(i, size_min=size_min, size_max=size_max, prompt_post_process=prior_loss_prompt_post_process)
            for i in prior_loss_train_data_dir
        ])))
    unet.train()
    current_model_index = 0
    while global_step <= max_train_steps:
        if len(pretrained_model_name_or_path) > 1 and global_step > 0 and global_step % swap_every_n_steps == 0:
            current_model_index = (current_model_index + 1) % len(pretrained_model_name_or_path)
            with 计时(accelerator, global_step, '相位转移', sync=True):
                相位转移(unet, unet_state_dicts[current_model_index])
        在训练正则化 = bool(prior_loss_train_data_dir and (global_step % prior_loss_steps == prior_loss_steps - 1))
        if 在训练正则化:
            batch = next(源p)
        else:
            batch = next(源)
        try:
            with 计时(accelerator, global_step, 'VAE', sync=True):
                model_input = vae_encode_with_cache(vae, batch["pixel_values"], cache_dir, batch["image_hash"]).to(weight_dtype)
            prompt_embeds, pooled_prompt_embeds = encode_prompt(batch['prompts'], compel)
        except Exception:
            logging.exception('准备输入时出了问题:')
            continue
        if global_step < 100:
            with open(f'{checkpoint_dir}/prompt_log.txt', 'a', encoding='utf8') as f:
                print('\n'.join(['-'*9, f'{在训练正则化=}', '【原本】', batch['raw_prompts'][0], '【改后】', batch['prompts'][0]]), file=f)
        with accelerator.accumulate(unet), 计时(accelerator, global_step, '全', sync=True):
            h, w = batch['pixel_values'].shape[2:]
            noise = torch.randn_like(model_input)
            bsz = model_input.shape[0]
            if 在训练正则化:
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device)
            else:
                timesteps = torch.randint(time_min, time_max, (bsz,), device=model_input.device).long()

            noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)    # 100接近model_input，800接近noise
            add_time_ids = torch.cat([compute_time_ids(s, r, c) for s, r, c in zip([(h, w)], [(h, w)], [(0, 0)])]).to('cuda')
            unet_added_conditions = {"time_ids": add_time_ids}
            unet_added_conditions.update({"text_embeds": pooled_prompt_embeds})

            model_pred = unet(
                noisy_model_input,
                timesteps,
                prompt_embeds,
                added_cond_kwargs=unet_added_conditions,
                return_dict=False,
            )[0]


            if 在训练正则化:
                with torch.inference_mode():
                    unet.disable_adapters()
                    原model_pred = unet(
                        noisy_model_input,
                        timesteps,
                        prompt_embeds,
                        added_cond_kwargs=unet_added_conditions,
                        return_dict=False,
                    )[0]
                    unet.enable_adapters()
                    target = 原model_pred.detach().clone()
            else:
                if prediction_type is not None:
                    noise_scheduler.register_to_config(prediction_type=prediction_type)
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(model_input, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            if snr_gamma is None:
                loss = conditional_loss(
                    model_pred.float(), target.float(), reduction="none", loss_type=loss_type, huber_c=huber_c
                )
                if use_mask and batch.get('pixel_values_mask') is not None:
                    x, y = batch["pixel_values_mask"].shape[-2:]
                    mask = F.interpolate(batch['pixel_values_mask'].to(accelerator.device, dtype=torch.float32), size=(x//8, y//8))
                    mask = mask * (1 - mask_min) + mask_min
                    loss = loss * mask
            else:
                # Compute loss-weights as per Section 3.4 of https://huggingface.co/papers/2303.09556.
                # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                # This is discussed in Section 4.2 of the same paper.
                snr = compute_snr(noise_scheduler, timesteps)
                mse_loss_weights = torch.stack([snr, snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0]
                if noise_scheduler.config.prediction_type == "epsilon":
                    mse_loss_weights = mse_loss_weights / snr
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    mse_loss_weights = mse_loss_weights / (snr + 1)
                loss = conditional_loss(
                    model_pred.float(), target.float(), reduction="none", loss_type=loss_type, huber_c=huber_c
                )
                if use_mask and batch.get('pixel_values_mask') is not None:
                    mask = F.interpolate(batch['pixel_values_mask'].to(accelerator.device, dtype=torch.float32), size=model_pred.shape[-2:])
                    mask = mask * (1 - mask_min) + mask_min
                    loss = loss * mask
                loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                grad_norm = accelerator.clip_grad_norm_([*filter(lambda p: p.requires_grad, unet.parameters())], max_grad_norm)
            with 计时(accelerator, global_step, 'optimizer', sync=True):
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

        if accelerator.sync_gradients:
            if global_step % validation_steps == 0 or global_step in [checkpointing_steps//2, checkpointing_steps//4]:
                validation(
                    accelerator=accelerator,
                    vae=vae,
                    text_encoder=unwrap_model(text_encoder_one),
                    text_encoder_2=unwrap_model(text_encoder_two),
                    unet=unwrap_model(unet),
                    torch_dtype=weight_dtype,
                    trackers=accelerator.trackers,
                    global_step=global_step,
                    pretrained_model_name_or_path=pretrained_model_name_or_path[0],
                    validation_prompt_list=validation_prompt_list,
                    current_model_index=current_model_index,
                )
            前缀 = '正则化' * 在训练正则化
            logs = {f"{前缀}loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], f"{前缀}grad_norm": grad_norm.item(), f"{前缀}t": timesteps[0]}
            for a, b in itertools.pairwise([1000, 900, 800, 0]):
                if a >= timesteps[0] >= b:
                    logs = logs | {
                        f'{前缀}grad_norm_{a}到{b}': grad_norm.item(),
                        f'{前缀}loss_{a}到{b}': loss.item(),
                    }
                    accelerator.log(logs, step=global_step)
                    break
            progress_bar.update(1)
            global_step += 1
            if accelerator.is_main_process and global_step % checkpointing_steps == 0:
                save_path = os.path.join(checkpoint_dir, f"checkpoint-{global_step}")
                accelerator.save_state(save_path)
                logging.info(f"Saved state to {save_path}")


# accelerate launch train通用.py --pretrained_model_name_or_path="R:\stable-diffusion-webui-master\models\Stable-diffusion\waiNSFWIllustrious_v100.safetensors" --train_data_dir="X:/ck3_loading_screens6" --output_dir="ck3" --batch_size=1 --validation_steps=500 --checkpointing_steps=500 --lr_warmup_steps=50 --gradient_checkpointing --seed=114514 --loss_type=huber --validation_prompt_list="1girl, twintails, white hair, school uniform, indoors;1girl, twintails, white hair, school uniform, outdoors;1girl, twintails, white hair, school uniform, outdoors, baram, starshadowmagician" --max_train_steps=5000
# "R:\stable-diffusion-webui-master\models\Stable-diffusion\waiNSFWIllustrious_v100.safetensors"
# "Y:\models\Diffusion\illustriousXL_v01.safetensors"
if __name__ == "__main__":
    fire.Fire(main)
