import io
import re
import random
import hashlib
from pathlib import Path

from PIL import Image
from torchvision import transforms


def _shuffle(s: str, start: int) -> str:
    a = [i.strip() for i in s.split(',')]
    a1, a2 = a[:start], a[start:]
    random.shuffle(a2)
    return ', '.join(a1 + a2)


def 读取数据集(p: str, *, drop_tag_rate=0, drop_text_rate=0, size_min, size_max, image_exts={'.jpg', '.jpeg', '.png', '.bmp', '.webp'}, prompt_post_process='', use_mask=False):
    def 处理图片(img, mask=None):
        w, h = img.size
        current_avg = (w + h) / 2.0
        target_avg = random.randint(size_min, size_max)
        scale_factor = target_avg / current_avg
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        crop_w, crop_h = new_w // 16 * 16, new_h // 16 * 16
        img = img.resize((new_w, new_h), Image.Resampling.BICUBIC).crop((0, 0, crop_w, crop_h))
        pixel_values = transforms.Normalize([0.5], [0.5])(transforms.ToTensor()(img))
        if mask is not None:
            mask = mask.resize((new_w, new_h), Image.Resampling.BICUBIC).crop((0, 0, crop_w, crop_h))
            pixel_values_mask = transforms.ToTensor()(mask)
            return pixel_values, pixel_values_mask
        return pixel_values

    a = []
    s = {*Path(p).iterdir()}
    for img_path in s:
        if img_path.suffix.lower() in image_exts and not img_path.stem.endswith('.mask'):
            txt_path = img_path.with_suffix('.txt')
            if txt_path in s:
                if use_mask:
                    mask_path, = [img_path.with_name(f'{img_path.stem}.mask{ext}') for ext in image_exts if img_path.with_name(f'{img_path.stem}.mask{ext}') in s]
                    a.append((img_path, txt_path, mask_path))
                else:
                    a.append((img_path, txt_path))
    print(f'{p}中找到了{len(a)}个对！')
    while True:
        random.shuffle(a)
        for item in a:
            img_path, txt_path = item[0], item[1]
            img_bytes = img_path.read_bytes()
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            if use_mask:
                mask = Image.open(item[2]).convert('L')
                pixel_values, pixel_values_mask = 处理图片(img, mask)
            else:
                pixel_values = 处理图片(img)
            s = txt_path.read_text(encoding='utf-8')
            if True:
                新sa = s.split(', ')
                if drop_tag_rate > 0:
                    for tag in 新sa:
                        if random.random() < drop_tag_rate:
                            新sa.remove(tag)
                新s = ', '.join(新sa)
                if drop_text_rate and random.random() < drop_text_rate:
                    新s = ''
            if prompt_post_process:
                新s = eval(prompt_post_process, {'s': 新s, 'random': random, 're': re, 'shuffle': _shuffle})
            res = {
                'pixel_values': pixel_values.unsqueeze(0),
                'prompts': [新s],
                'raw_prompts': [s],
                'image_hash': hashlib.sha256(img_bytes).hexdigest()[:8]+'_'.join(map(str, pixel_values.shape)),
            }
            if use_mask:
                res['pixel_values_mask'] = pixel_values_mask.unsqueeze(0)
            yield res
