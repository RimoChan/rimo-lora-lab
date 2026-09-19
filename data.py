import io
import re
import json
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

def _drop(s: str, rate: float, protect: list[str]) -> str:
    tags = [i.strip() for i in s.split(',') if i.strip()]
    保留 = [i for i in tags if i in protect or random.random() > rate]
    return ', '.join(保留)

def _置换(s: str) -> str:
    return ', '.join(s.split(' ')).replace('_', ' ')

def _人数标签(s: str) -> bool:
    if s in ('1girl', '1boy'):
        return True
    return bool(re.fullmatch('[0-9]\+?(girl|boy)s$', s))

def _分离人数标签(原始tags: list[str]) -> tuple[list[str], list[str]]:
    人 = [i for i in 原始tags if _人数标签(i)]
    剩下的 = [i for i in 原始tags if not _人数标签(i)]
    return 人, 剩下的

def _balance_drop_rate(s: str, tag次数: dict, balance_rate: str) -> float:
    包含tags = [tag for tag in tag次数 if tag in s]
    if 包含tags:
        return (1 - min(tag次数.values()) / min(tag次数[tag] for tag in 包含tags)) * balance_rate
    return 0

def _read(txt_path: Path, r) -> str:
    s = txt_path.read_text(encoding='utf-8')
    if txt_path.suffix == '.json':
        d = json.loads(s)
        q, w = _分离人数标签(d['tag_string_general'].split(' '))
        r.shuffle(w)
        s = _置换(' '.join([*q, d['tag_string_character'], *w, d['tag_string_artist']]))
    return s

def 读取数据集(p: str, *, size_min, size_max, balance_tags=None, balance_rate=1.0, image_exts={'.jpg', '.jpeg', '.png', '.bmp', '.webp'}, prompt_post_process='', use_mask=False, seed=1, ep=10**8):
    r = random.Random(seed)

    def 处理图片(img, mask=None):
        w, h = img.size
        current_avg = (w + h) / 2.0
        target_avg = r.randint(size_min, size_max)
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
            t = [img_path.with_suffix(ext) for ext in ('.txt', '.json') if img_path.with_suffix(ext) in s]
            if t:
                txt_path = t[0]
                if use_mask:
                    mask_path, = [img_path.with_name(f'{img_path.stem}.mask{ext}') for ext in image_exts if img_path.with_name(f'{img_path.stem}.mask{ext}') in s]
                    a.append((img_path, txt_path, mask_path))
                else:
                    a.append((img_path, txt_path))
    assert a, f'{p}中没有数据！'
    a = sorted(a)
    print(f'{p}中找到了{len(a)}个对！')
    if balance_tags:
        tag次数 = {tag: 0 for tag in balance_tags}
        for item in a:
            txt = _read(item[1], r)
            for tag in balance_tags:
                if tag in txt:
                    tag次数[tag] += 1
        print('均衡发现', tag次数)
    for _ in range(ep):
        r.shuffle(a)
        for item in a:
            img_path, txt_path = item[0], item[1]
            s = _read(txt_path, r)
            if balance_tags:
                if r.random() < _balance_drop_rate(s, tag次数, balance_rate):
                    continue
            img_bytes = img_path.read_bytes()
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            if use_mask:
                mask = Image.open(item[2]).convert('L')
                pixel_values, pixel_values_mask = 处理图片(img, mask)
            else:
                pixel_values = 处理图片(img)
            if prompt_post_process:
                新s = eval(prompt_post_process, {'s': s, 'random': random, 're': re, 'shuffle': _shuffle, 'drop': _drop})
            else:
                新s = s
            res = {
                'pixel_values': pixel_values.unsqueeze(0),
                'prompts': [新s],
                'raw_prompts': [s],
                'image_hash': hashlib.sha256(img_bytes).hexdigest()[:8]+'_'.join(map(str, pixel_values.shape)),
            }
            if use_mask:
                res['pixel_values_mask'] = pixel_values_mask.unsqueeze(0)
            yield res
