import io
import random
import hashlib
from pathlib import Path

from PIL import Image
from torchvision import transforms


def add(a, b):
    return a + b


def 读取数据集(p: str, *, shuffle_tag=True, drop_tag_rate=0, drop_text_rate=0, size_min, size_max, image_exts={'.jpg', '.jpeg', '.png', '.bmp', '.webp'}, prompt_post_process='', prompt_post_process_arg=None):
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
    s = {*Path(p).iterdir()}
    for img_path in s:
        if img_path.suffix.lower() in image_exts:
            txt_path = img_path.with_suffix('.txt')
            # if txt_path.exists():
            if txt_path in s:
                a.append((img_path, txt_path))
    print(f'{p}中找到了{len(a)}个对！')
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
                    for tag in 新sa:
                        if random.random() < drop_tag_rate:
                            新sa.remove(tag)
                if shuffle_tag:
                    random.shuffle(新sa)
                新s = ', '.join(新sa)
                if drop_text_rate and random.random() < drop_text_rate:
                    新s = ''
            if prompt_post_process:
                f = globals()[prompt_post_process]
                新s = f(新s, prompt_post_process_arg)
            yield {
                'pixel_values': pixel_values.unsqueeze(0),
                'prompts': [新s],
                'raw_prompts': [s],
                'image_hash': hashlib.sha256(img_bytes).hexdigest()[:8]+'_'.join(map(str,pixel_values.shape)),
            }
