import os
import math
import time
import logging
import datetime
import fire
from pathlib import Path

import requests
import safetensors.torch
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from datasets import load_dataset
from packaging import version
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from tqdm.auto import tqdm

import diffusers
from diffusers import DDPMScheduler, StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from diffusers.loaders import StableDiffusionLoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params, compute_snr
from diffusers.utils import (
    convert_state_dict_to_diffusers,
    convert_unet_state_dict_to_peft,
)
from diffusers.utils.torch_utils import is_compiled_module

from compel import Compel, ReturnedEmbeddingsType
from dan后处理 import dan后处理
from common import cycle, clean, 生成optimizer, 哈


logger = get_logger(__name__)


arg_validation_prompt = [
    ("1girl, momoi (blue archive), typing on keyboard, computer, sitting, angry, indoors, fuzichoco", 100),
    ("1girl, momoi (blue archive), oversplit, ballerina, fuzichoco", 100),
    ("1girl, azusa (blue archive), arabesque (pose), ballerina, momoko (momopoco)", 102),
    ("1girl, mari (blue archive), retire (ballet), indoors, ballerina, baram, starshadowmagician", 104),
]


def gen_dataset(accelerator, train_data_dir, regular_train_data_dir, drop_tag_rate) -> tuple:
    d后 = dan后处理(drop_tag_rate=0, drop_char_feature_rate=0, size=(576, 1344), 新drop_tag_rate=drop_tag_rate, 新drop_tag保留=['retire_(ballet)', 'arabesque_(pose)', 'oversplit'])
    d1 = load_dataset(
        "imagefolder",
        data_files={"train": os.path.join(train_data_dir, "**")},
    )
    with accelerator.main_process_first():
        train_dataset = d1["train"].with_transform(d后.preprocess_train, output_all_columns=True)
    d2 = load_dataset(
        "imagefolder",
        data_files={'正则化': os.path.join(regular_train_data_dir, "**")},
    )
    with accelerator.main_process_first():
        正则化dataset = d2["正则化"].with_transform(d后.preprocess_train, output_all_columns=True)
    return train_dataset, 正则化dataset


def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


def encode_prompt(prompt_batch, compel) -> tuple:
    captions = []
    for caption in prompt_batch:
        assert isinstance(caption, str)
        captions.append(caption)

    with torch.no_grad():
        try:
            t = compel(captions)
        except RuntimeError as e:
            if 'Sizes of tensors must match' in str(e):     # Compel本来就会padding，但是偶尔还是有这个错，很奇怪，先跳过试试
                t = compel(['' for i in captions])
            else:
                raise
        return t


# def conditional_loss(model_pred: torch.Tensor, target: torch.Tensor, reduction: str = "mean", loss_type: str = "l2", huber_c: float = 0.1,):
#     if loss_type == "l2":
#         loss = F.mse_loss(model_pred, target, reduction=reduction)
#     elif loss_type == "huber" or loss_type == "huber_scheduled":
#         loss = huber_c * (torch.sqrt((model_pred - target) ** 2 + huber_c**2) - huber_c)
#         if reduction == "mean":
#             loss = torch.mean(loss)
#         elif reduction == "sum":
#             loss = torch.sum(loss)
#     else:
#         raise NotImplementedError(f"Unsupported Loss Type {loss_type}")
#     return loss


def 检查模型类型(path) -> str:
    结果 = []
    m = {
        'model.diffusion_model.output_blocks.4.1.transformer_blocks.1.ff.net.0.proj.bias': 'sdxl',
        'model.diffusion_model.output_blocks.9.1.transformer_blocks.0.norm3.bias': 'sd1.5',
    }
    for k in safetensors.torch.load_file(path).keys():
        if k in m:
            结果.append(m[k])
    assert len(结果) == 1
    return 结果[0]


def validation(global_step, accelerator, vae, text_encoder, text_encoder_2, unet, torch_dtype, trackers, pretrained_model_name_or_path):
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
            clean()
            images = [
                pipeline(
                    prompt=prompt,
                    generator=torch.Generator(device=accelerator.device).manual_seed(seed),
                    num_inference_steps=20,
                    guidance_scale=7,
                    width=768,
                    height=1024,
                ).images[0] for prompt, seed in arg_validation_prompt
            ]
            clean()
            for tracker in trackers:
                if tracker.name == "tensorboard":
                    np_images = np.stack([np.asarray(img) for img in images])
                    tracker.writer.add_images("validation", np_images, global_step, dataformats="NHWC")
            del pipeline


def 加载模型(path) -> tuple:
    模型类型 = 检查模型类型(path)
    assert 模型类型 == 'sdxl'
    临时pipeline = StableDiffusionXLPipeline.from_single_file(path)
    text_encoder_one = 临时pipeline.text_encoder
    text_encoder_two = 临时pipeline.text_encoder_2
    compel = Compel(truncate_long_prompts=False, tokenizer=[临时pipeline.tokenizer, 临时pipeline.tokenizer_2], text_encoder=[text_encoder_one, text_encoder_two],  returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, requires_pooled=[False, True])
    vae = 临时pipeline.vae
    unet = 临时pipeline.unet
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    unet.requires_grad_(False)
    noise_scheduler = DDPMScheduler.from_config(临时pipeline.scheduler.config)
    return text_encoder_one, text_encoder_two, compel, unet, vae, noise_scheduler


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

    # q = unet.state_dict()[检查]
    # qq = dd[检查]
    # print('转移啊', q.abs().mean(), q.var(), qq.abs().mean(), qq.var())

    unet.load_state_dict(dd, strict=False)

检查 = 'down_blocks.2.attentions.0.transformer_blocks.0.attn2.to_v.base_layer.weight'

def main(
    pretrained_model_name_or_path: str,
    pretrained_cross_model_path: str = None,
    dataset_config_name: str = None,
    train_data_dir: str = None,
    regular_train_data_dir: str = None,
    validation_steps: int = 100,
    output_dir: str = "sd-model-finetuned-lora",
    cache_dir: str = None,
    seed: int = None,
    train_batch_size: int = 16,
    num_train_epochs: int = 100,
    max_train_steps: int = None,
    checkpointing_steps: int = 500,
    resume_from_checkpoint: str = None,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = False,
    learning_rate: float = 1e-4,
    scale_lr: bool = False,
    lr_scheduler: str = "constant",
    lr_warmup_steps: int = 500,
    snr_gamma: float = None,
    dataloader_num_workers: int = 0,
    optimizer: str = "adam",
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-8,
    max_grad_norm: float = 1.0,
    hub_token: str = None,
    prediction_type: str = None,
    logging_dir: str = "logs",
    mixed_precision: str = None,
    enable_xformers_memory_efficient_attention: bool = False,
    noise_offset: float = 0,
    loss_type: str = "l2",
    huber_schedule: str = "snr",
    huber_c: float = 0.1,
    rank: int = 64,
    alpha: int = 32,
    time_min: int = 0,
    time_max: int = 1000,
    drop_tag_rate: float = 0.0,
):
    logging_dir = Path(output_dir, logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with='tensorboard',
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if seed is not None:
        set_seed(seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    text_encoder_one, text_encoder_two, compel, unet, vae, noise_scheduler = 加载模型(pretrained_model_name_or_path)
    if pretrained_cross_model_path:
        _, __, ___, unet_b, ____, _____ = 加载模型(pretrained_cross_model_path)
        unet_a_state_dict_cpu = unet.to('cpu', dtype=weight_dtype).state_dict()
        unet_b_state_dict_cpu = unet_b.to('cpu', dtype=weight_dtype).state_dict()
        del unet_b

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
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

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            # there are only two options here. Either are just the unet attn processor layers
            # or there are the unet and text encoder attn layers
            unet_lora_layers_to_save = None
            text_encoder_one_lora_layers_to_save = None
            text_encoder_two_lora_layers_to_save = None

            for model in models:
                if isinstance(unwrap_model(model), type(unwrap_model(unet))):
                    unet_lora_layers_to_save = convert_state_dict_to_diffusers(get_peft_model_state_dict(model))
                elif isinstance(unwrap_model(model), type(unwrap_model(text_encoder_one))):
                    text_encoder_one_lora_layers_to_save = convert_state_dict_to_diffusers(
                        get_peft_model_state_dict(model)
                    )
                elif isinstance(unwrap_model(model), type(unwrap_model(text_encoder_two))):
                    text_encoder_two_lora_layers_to_save = convert_state_dict_to_diffusers(
                        get_peft_model_state_dict(model)
                    )
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                if weights:
                    weights.pop()

            StableDiffusionXLPipeline.save_lora_weights(
                output_dir,
                unet_lora_layers=unet_lora_layers_to_save,
                text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
                text_encoder_2_lora_layers=text_encoder_two_lora_layers_to_save,
            )

    def load_model_hook(models, input_dir):
        unet_ = None
        text_encoder_one_ = None
        text_encoder_two_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(unwrap_model(unet))):
                unet_ = model
            elif isinstance(model, type(unwrap_model(text_encoder_one))):
                text_encoder_one_ = model
            elif isinstance(model, type(unwrap_model(text_encoder_two))):
                text_encoder_two_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict, _ = StableDiffusionLoraLoaderMixin.lora_state_dict(input_dir)
        unet_state_dict = {f"{k.replace('unet.', '')}": v for k, v in lora_state_dict.items() if k.startswith("unet.")}
        unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
        incompatible_keys = set_peft_model_state_dict(unet_, unet_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

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

    if scale_lr:
        learning_rate = (
            learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if mixed_precision == "fp16":
        models = [unet]
        cast_training_params(models, dtype=torch.float32)

    optimizer = 生成optimizer(optimizer, unet, adam_beta1, adam_beta2, adam_weight_decay, adam_epsilon, learning_rate, learning_rate * 20)

    train_dataset, 正则化dataset = gen_dataset(accelerator, train_data_dir, regular_train_data_dir, drop_tag_rate)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        original_sizes = [example["original_sizes"] for example in examples]
        resized_sizes = [example["resized_sizes"] for example in examples]
        crop_top_lefts = [example["crop_top_lefts"] for example in examples]
        prompts = [example["prompts"] for example in examples]
        result = {
            "pixel_values": pixel_values,
            "prompts": prompts,
            "original_sizes": original_sizes,
            "resized_sizes": resized_sizes,
            "crop_top_lefts": crop_top_lefts,
        }

        filenames = [example["filenames"] for example in examples if "filenames" in example]
        if filenames:
            result["filenames"] = filenames
        return result

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=train_batch_size,
        num_workers=dataloader_num_workers,
    )
    正则化dataloader = torch.utils.data.DataLoader(
        正则化dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=train_batch_size,
        num_workers=dataloader_num_workers,
    )

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    if max_train_steps is None:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
        num_cycles=10,
    )

    unet, optimizer, train_dataloader, 正则化dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, 正则化dataloader, lr_scheduler
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    if overrode_max_train_steps:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    特征 = f'{哈(train_data_dir)}-{哈(pretrained_model_name_or_path)}-{optimizer}-' + f'snr{snr_gamma}'*bool(snr_gamma) + f'-{loss_type}-lr{learning_rate}-drop{drop_tag_rate}-{mixed_precision}-{lr_scheduler}-lora{rank}_{alpha}-time{time_min}_{time_max}_' + f'-X_{哈(pretrained_cross_model_path)}' * bool(pretrained_cross_model_path)
    if accelerator.is_main_process:
        时间 = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
        存档名 = f'{时间}-' + 特征
        accelerator.init_trackers(存档名, config={
            "pretrained_model_name_or_path": pretrained_model_name_or_path,
            "pretrained_cross_model_path": pretrained_cross_model_path,
            "dataset_config_name": dataset_config_name,
            "train_data_dir": train_data_dir,
            "regular_train_data_dir": regular_train_data_dir,
            "validation_steps": validation_steps,
            "output_dir": output_dir,
            "cache_dir": cache_dir,
            "seed": seed,
            "train_batch_size": train_batch_size,
            "num_train_epochs": num_train_epochs,
            "max_train_steps": max_train_steps,
            "checkpointing_steps": checkpointing_steps,
            "resume_from_checkpoint": resume_from_checkpoint,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "gradient_checkpointing": gradient_checkpointing,
            "learning_rate": learning_rate,
            "scale_lr": scale_lr,
            "lr_scheduler": lr_scheduler,
            "lr_warmup_steps": lr_warmup_steps,
            "snr_gamma": snr_gamma,
            "dataloader_num_workers": dataloader_num_workers,
            "optimizer": optimizer,
            "adam_beta1": adam_beta1,
            "adam_beta2": adam_beta2,
            "adam_weight_decay": adam_weight_decay,
            "adam_epsilon": adam_epsilon,
            "max_grad_norm": max_grad_norm,
            "hub_token": hub_token,
            "prediction_type": prediction_type,
            "logging_dir": logging_dir,
            "mixed_precision": mixed_precision,
            "enable_xformers_memory_efficient_attention": enable_xformers_memory_efficient_attention,
            "noise_offset": noise_offset,
            "loss_type": loss_type,
            "huber_schedule": huber_schedule,
            "huber_c": huber_c,
            "rank": rank,
            "alpha": alpha,
            "time_min": time_min,
            "time_max": time_max,
            "drop_tag_rate": drop_tag_rate,
        })
    checkpoint_dir = os.path.join(output_dir, 特征)

    # Train!
    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0

    initial_global_step = 0
    if resume_from_checkpoint:
        if resume_from_checkpoint != "latest":
            path = os.path.basename(resume_from_checkpoint)
        else:
            dirs = os.listdir(output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        assert path, f"Checkpoint '{resume_from_checkpoint}' does not exist"
        accelerator.print(f"Resuming from checkpoint {path}")
        accelerator.load_state(os.path.join(output_dir, path))
        global_step = int(path.split("-")[1])
        initial_global_step = global_step

    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    正则化dataloader_cycle = cycle(正则化dataloader)
    train_dataloader_cycle = cycle(train_dataloader)

    unet.train()
    train_loss = 0.0
    swap_every_n_steps = 8
    current_model_is_a = True
    while global_step <= max_train_steps:
        if pretrained_cross_model_path and global_step > 0 and global_step % swap_every_n_steps == 0:
            base_model_to_swap = unet
            if current_model_is_a:
                相位转移(base_model_to_swap, unet_b_state_dict_cpu)
            else:
                相位转移(base_model_to_swap, unet_a_state_dict_cpu)
            current_model_is_a = not current_model_is_a
        在训练正则化 = global_step % 4 == 3
        if 在训练正则化:
            batch = next(正则化dataloader_cycle)
        else:
            batch = next(train_dataloader_cycle)
        with accelerator.accumulate(unet):
            # Convert images to latent space
            pixel_values = batch["pixel_values"]

            model_input = vae.encode(pixel_values).latent_dist.sample()
            model_input = model_input * vae.config.scaling_factor
            model_input = model_input.to(weight_dtype)

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(model_input)
            if noise_offset:
                # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                noise += noise_offset * torch.randn(
                    (model_input.shape[0], model_input.shape[1], 1, 1), device=model_input.device
                )

            bsz = model_input.shape[0]
            if 在训练正则化:
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device)
            else:
                timesteps = torch.randint(time_min, time_max, (bsz,), device=model_input.device)

            huber_c = 1

            timesteps = timesteps.long()

            # Add noise to the model input according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)
            # 100接近model_input，800接近noise

            def compute_time_ids(original_size, resized_size, crops_coords_top_left):
                # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
                target_size = resized_size
                add_time_ids = list(original_size + crops_coords_top_left + target_size)
                add_time_ids = torch.tensor([add_time_ids])
                add_time_ids = add_time_ids.to(accelerator.device, dtype=weight_dtype)
                return add_time_ids

            add_time_ids = torch.cat(
                [compute_time_ids(s, r, c) for s, r, c in zip(batch["original_sizes"], batch["resized_sizes"], batch["crop_top_lefts"])]
            )
            prompt_embeds, pooled_prompt_embeds = encode_prompt(batch['prompts'], compel)
            unet_added_conditions = {"time_ids": add_time_ids}
            unet_added_conditions.update({"text_embeds": pooled_prompt_embeds})

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
            model_pred = unet(
                noisy_model_input,
                timesteps,
                prompt_embeds,
                added_cond_kwargs=unet_added_conditions,
                return_dict=False,
            )[0]

            if 在训练正则化:
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
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            else:
                # Compute loss-weights as per Section 3.4 of https://huggingface.co/papers/2303.09556.
                # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                # This is discussed in Section 4.2 of the same paper.
                snr = compute_snr(noise_scheduler, timesteps)
                mse_loss_weights = torch.stack([snr, snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                    dim=1
                )[0]
                if noise_scheduler.config.prediction_type == "epsilon":
                    mse_loss_weights = mse_loss_weights / snr
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    mse_loss_weights = mse_loss_weights / (snr + 1)
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                loss = loss.mean()

            # Gather the losses across all processes for logging (if we use distributed training).
            avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()
            train_loss += avg_loss.item() / gradient_accumulation_steps

            # Backpropagate
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                grad_norm = accelerator.clip_grad_norm_([*filter(lambda p: p.requires_grad, unet.parameters())], max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        if arg_validation_prompt is not None and global_step % validation_steps == 0:
            validation(
                accelerator=accelerator,
                vae=vae,
                text_encoder=unwrap_model(text_encoder_one),
                text_encoder_2=unwrap_model(text_encoder_two),
                unet=unwrap_model(unet),
                torch_dtype=weight_dtype,
                trackers=accelerator.trackers,
                global_step=global_step,
                pretrained_model_name_or_path=pretrained_model_name_or_path,
            )

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            progress_bar.update(1)
            global_step += 1
            if 在训练正则化:
                accelerator.log({"正则化loss": train_loss, "正则化t": timesteps[0], "正则化grad_norm": grad_norm.item()}, step=global_step)
            else:
                accelerator.log({"train_loss": train_loss, "t": timesteps[0], "grad_norm": grad_norm.item()}, step=global_step)
            accelerator.log({"input_width": batch["resized_sizes"][0][0]}, step=global_step)
            train_loss = 0.0

            if accelerator.is_main_process:
                if global_step % checkpointing_steps == 0:
                    save_path = os.path.join(checkpoint_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

        logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
        progress_bar.set_postfix(**logs)

        if global_step >= max_train_steps:
            break

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unwrap_model(unet)
        unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))

        text_encoder_lora_layers = None
        text_encoder_2_lora_layers = None

        StableDiffusionXLPipeline.save_lora_weights(
            save_directory=output_dir,
            unet_lora_layers=unet_lora_state_dict,
            text_encoder_lora_layers=text_encoder_lora_layers,
            text_encoder_2_lora_layers=text_encoder_2_lora_layers,
        )

        del unet
        del text_encoder_one
        del text_encoder_two
        del text_encoder_lora_layers
        del text_encoder_2_lora_layers
        torch.cuda.empty_cache()

    accelerator.end_training()


if __name__ == "__main__":
    fire.Fire(main)
