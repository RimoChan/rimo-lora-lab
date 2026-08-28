from pathlib import Path

from tqdm import tqdm
import cv2
import numpy as np
from imgutils.segment import get_isnetis_mask


文件夹 = './'

a = [i for i in Path(文件夹).iterdir() if not str(i).endswith('.mask.jpg')]
for i in tqdm(a):
    if i.suffix in {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}:
        mask = get_isnetis_mask(i)
        h, w = mask.shape
        膨胀半径 = int(max(h, w) * 0.08)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (膨胀半径 * 2 + 1, 膨胀半径 * 2 + 1))
        二值mask = (mask > 0.5).astype(np.uint8) * 255
        膨胀mask = cv2.dilate(二值mask, kernel)
        cv2.imencode('.jpg', 膨胀mask)[1].tofile(f'{文件夹}/{i.stem}.mask.jpg')
