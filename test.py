import os
import glob
import shutil
import subprocess

import fire

from common import 注入cl

注入cl()


def main():
    output_uncompiled = '_测试不compile'
    output_compiled: str = '_测试compile'

    for i in (output_uncompiled, output_compiled):
        shutil.rmtree(i, ignore_errors=True)

    subprocess.run(f'accelerate launch train.py --config=./test/test.json --output_dir=test/{output_uncompiled} --mixed_precision=bf16 --use_compile=False', check=True)
    subprocess.run(f'accelerate launch train.py --config=./test/test.json --output_dir=test/{output_compiled} --mixed_precision=bf16 --use_compile=True', check=True)

    print('='*50)
    print(glob.glob(os.path.join('test', output_uncompiled, '**', 'loss_log.txt'), recursive=True))
    print(glob.glob(os.path.join('test', output_compiled, '**', 'loss_log.txt'), recursive=True))


if __name__ == '__main__':
    fire.Fire(main)
