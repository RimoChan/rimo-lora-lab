import gc
import io
import math
import time
import random
import hashlib
import threading
import contextlib
from queue import Queue
from pathlib import Path
from typing import Optional

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Optimizer
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from tensorboard.compat.proto import summary_pb2
import safetensors.torch


def buffered_iterator(iterator, maxsize=10):
    q = Queue(maxsize=maxsize)
    sentinel = object()

    def worker():
        for item in iterator:
            q.put(item)
        q.put(sentinel)

    threading.Thread(target=worker, daemon=True).start()
    yield from iter(q.get, sentinel)


@contextlib.contextmanager
def 计时(accelerator, global_step, 名字, sync=False):
    if sync and accelerator.is_main_process:
        torch.cuda.synchronize()
    开始时间 = time.time()
    yield
    if accelerator.is_main_process:
        torch.cuda.synchronize()
        accelerator.log({f'【计时】{名字}': time.time() - 开始时间}, step=global_step)


def 哈(x) -> str:
    return hashlib.md5(str(x).encode()).hexdigest().upper()[:3]


def cycle(iterable_obj):
    while True:
        yield from iterable_obj


def clean():
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()


def clone_state_dict(d: dict) -> dict: 
    new_state_dict = {}
    for key, value in d.items():
        new_state_dict[key] = value.detach().clone()
    return new_state_dict


def compute_time_ids(original_size, resized_size, crops_coords_top_left):
    # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
    target_size = resized_size
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    add_time_ids = torch.tensor([add_time_ids])
    return add_time_ids


def is_muon(name, param):
    skip_keys = ["embed_tokens", "lm_head", "tok_embeddings", "output"]
    return param.ndim >= 2 and not any(key in name for key in skip_keys)


def encode_prompt(prompt_batch, compel) -> tuple:
    captions = []
    for caption in prompt_batch:
        assert isinstance(caption, str)
        captions.append(caption)
    return compel(captions)


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


def 生成optimizer(args_optimizer, unet, adam_beta1, adam_beta2, adam_weight_decay, adam_epsilon, learning_rate, learning_rate_muon):
    import torch
    if 'adam' in args_optimizer:
        if args_optimizer == 'adam':
            optimizer_class = torch.optim.AdamW
        elif args_optimizer == '8bit_adam':
            import bitsandbytes as bnb
            optimizer_class = bnb.optim.AdamW8bit
        params_to_optimize = [{
            "params": unet.parameters(),
            "lr": learning_rate,
        }]
        optimizer = optimizer_class(
            params_to_optimize,
            betas=(adam_beta1, adam_beta2),
            weight_decay=adam_weight_decay,
            eps=adam_epsilon,
        )
    elif args_optimizer == 'prodigy':
        from prodigyopt import Prodigy
        params_to_optimize = unet.parameters()
        optimizer = Prodigy(params_to_optimize, lr=1., weight_decay=0.01, slice_p=11, safeguard_warmup=True, use_bias_correction=True)
    elif args_optimizer == 'muon':
        from muon import SingleDeviceMuonWithAuxAdam
        hidden_weights = [p for k, p in unet.named_parameters() if is_muon(k, p) and p.requires_grad]
        hidden_gains_biases = [p for k, p in unet.named_parameters() if not is_muon(k, p) and p.requires_grad]
        param_groups = [
            dict(params=hidden_weights, use_muon=True, lr=learning_rate_muon, weight_decay=0.01),
            dict(params=hidden_gains_biases, use_muon=False, lr=learning_rate, betas=(adam_beta1, adam_beta2), weight_decay=adam_weight_decay),
        ]
        optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
    else:
        raise ValueError(f'这是什么优化器？{args_optimizer}')
    return optimizer


def 评测pipeline(pipe, n_iter, tags_seed=0, random_seed=0, guidance_scale_range=(6, 8)):
    import benchmarker.ml_danbooru
    from benchmarker.common import 要测的标签, ml_danbooru标签2
    rd_tag = random.Random(tags_seed)
    rd = random.Random(random_seed)
    所有得分 = []
    for _ in tqdm(range(n_iter), desc='评测'):
        标签个数 = rd.randint(16, 21)
        标签组 = rd_tag.sample(要测的标签, 标签个数)
        assert '_' not in str(标签组)
        下划线标签组 = [i.strip().replace(' ', '_') for i in 标签组]
        images = pipe(
            prompt=f'1 girl, {", ".join(标签组)}',
            negative_prompt = rd.choice(['worst quality, low quality', 'worst quality, low quality, blurry, greyscale, monochrome']),
            generator=torch.Generator(device='cuda').manual_seed(rd.randint(0, 2**16)),
            num_inference_steps=18+rd.randint(0, 6),
            guidance_scale=random.randint(*guidance_scale_range),
            width=704+rd.randint(0, 5)*64,
            height=704+rd.randint(0, 5)*64,
        ).images
        预测标签 = ml_danbooru标签2(images)[0]
        得分 = len(set(下划线标签组) & set(预测标签)) / len(set(下划线标签组))
        所有得分.append(得分)
    benchmarker.ml_danbooru.model = None
    return sum(所有得分) / len(所有得分)


def add_image_jpeg(writer, tag, img, global_step, quality=90):
    img_pil = Image.fromarray(img)
    output = io.BytesIO()
    img_pil.save(output, format='JPEG', quality=quality)
    img_str = output.getvalue()
    img_proto = summary_pb2.Summary.Image(
        height=img_pil.height,
        width=img_pil.width,
        colorspace=3,
        encoded_image_string=img_str
    )
    summary = summary_pb2.Summary(value=[
        summary_pb2.Summary.Value(tag=tag, image=img_proto)
    ])
    writer.file_writer.add_summary(summary, global_step)


def get_cosine_with_hard_restarts_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: int = 1, last_epoch: int = -1
) -> LambdaLR:
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        if progress >= 1.0:
            return 0.0
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0)))) * 0.9 + 0.1

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def cosine_with_restart_scheduler改(
    optimizer: Optimizer,
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
    num_cycles: int = 1,
    last_epoch: int = -1,
) -> LambdaLR:
    return get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        last_epoch=last_epoch,
    )


def 读取数据集(p: str, drop_tag_rate, drop_text_rate, size_min, size_max, image_exts={'.jpg', '.jpeg', '.png', '.bmp', '.webp'}):
    def random_scale(img):
        w, h = img.size
        current_avg = (w + h) / 2.0
        target_avg = random.randint(size_min, size_max)
        scale_factor = target_avg / current_avg
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        return img.resize((new_w, new_h), Image.Resampling.BICUBIC)

    transform = transforms.Compose([
        transforms.Lambda(random_scale),
        transforms.Lambda(lambda x: x.crop((0, 0, x.width // 16 * 16, x.height // 16 * 16))),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    a = []
    for img_path in Path(p).iterdir():
        if img_path.suffix.lower() in image_exts:
            txt_path = img_path.with_suffix('.txt')
            if txt_path.exists():
                a.append((img_path, txt_path))
    print(f'找到了{len(a)}个对！')
    while True:
        random.shuffle(a)
        for img_path, txt_path in a:
            img_bytes = img_path.read_bytes()
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            pixel_values = transform(img)
            s = txt_path.read_text(encoding='utf-8')
            if True:
                新sa = s.split(', ')
                if drop_tag_rate > 0:
                    新sa = random.sample(新sa, round(len(新sa) * (1 - drop_tag_rate)))
                random.shuffle(新sa)
                新s = ', '.join(新sa)
                if random.random() < drop_text_rate:
                    新s = ''
            yield {
                'pixel_values': pixel_values.unsqueeze(0),
                'prompts': [新s],
                'image_hash': hashlib.sha256(img_bytes).hexdigest()[:8]+'_'.join(map(str,pixel_values.shape)),
            }
