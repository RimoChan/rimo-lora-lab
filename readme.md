# 莉沫酱LoRA实验室！

大家平时会自己训练扩散模型的LoRA吗？

我有时候会想到1些新的力量<sub>(?)</sub>来提高Lora的训练效果，但是去fork别人的仓库来改又太笨重了，所以就自己写了1个！

不过现在还只支持SDXL，Anima之类的就以后再加吧！


## 加了什么

### 在多个base model上训练Lora

首先是这个。大家经常会担心自己训练的Lora在别的base model上效果不好，那与其说我想1堆办法去增加泛化性，不如直接在多个base model上1起训！

实现方式也非常简单，直接把所有的模型都放在RAM里，然后隔8个step把它们轮流传输到VRAM。

但也有观众可能会想，那怎么证明它不是只适应了训练时的那几个模型呢？

嗯……注，有人托梦给我！


### 蒸馏的正则化

然后是正则化，原本应该是叫先验损失，正则化这个叫法是哪里来的，总之字短1点我也就跟着叫了……

原本DreamBooth的那个实现，我是感觉既然都训LoRA了，它其实是比较不合适的，所以这个地方调整了1下，这里的实现改成了对于任意的Xt，用Lora推理1次，然后卸载Lora再推理1次，把2次的结果的差用作loss。

好处是正则化的方向更稳定了，可以保证在初始化的时候先验损失1定是0。

而且流程也比较简单，不用提前生成正则化图片。甚至还可以用同1个训练集，把触发词去掉直接正则自己，效果也很好。


### 假面！

添加了1个mask的功能！

比如想要训练某个角色，但是又不想把背景也学进去的话，就可以用mask把背景的部分涂掉，这部分像素就不会算进loss了。

还有我突然想到一件事，但如果1个mask不是2值的话，那它岂不就成了veil！


### Timestep区间

训练画风的时候，会担心画师的构图不小心被学到，或者训练动作的时候，也不希望模型去学细节。

所以就加上了这个限制训练时的timestep的功能，这样就可以自定义要丢掉高噪部分或是低噪部分了。


## 环境准备

这个代码应该不太需要特定的版本，可以先跑着，遇到什么没装再从`requirements.txt`里面挑出来装。

因为我也很难确定依赖的上下限，所以写的都是等号，直接`pip install -r requirements.txt`可能会把原本的环境给覆盖掉。


## 参数

启动脚本的话，就是这样，我们下面来介绍具体的参数:

```bash
accelerate launch train.py \
  --pretrained_model_name_or_path="models/illustriousXL_v01.safetensors" \
  --train_data_dir="dataset/my_character" \
  --output_dir="lora_output" \
  --lr=1e-4 \
  --rank=32 \
  --validation_steps=200 \
  --validation_prompt_list="1girl, solo, masterpiece;1girl, outdoors, sunset" \
  --max_train_steps=3000
```

### 基础与路径参数
| 参数 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `pretrained_model_name_or_path` | `str` | - | 底模路径 |
| `train_data_dir` | `str` | - | 训练数据集路径 |
| `prior_loss_train_data_dir` | `str` | `None` | 正则化数据集目录，没有就不会做 |
| `output_dir` | `str` | `'lora'` | 训练产物输出目录 |
| `cache_dir` | `str` | `'./rimo_lora_lab_cache'` | VAE Latent缓存目录 |
| `resume_from_checkpoint` | `str` | `'latest'` | 恢复训练的checkpoint路径 |
| `validation_prompt_list` | `str` | `list` | `'1girl'` | 验证的prompt |

- `pretrained_model_name_or_path`、`train_data_dir`、`prior_loss_train_data_dir`、`validation_prompt_list`，这几个参数，它们都可以填多个，写法是还是1个字符串，但是中间用 `;` 分隔。
  - 但是base_model实际上只会交换unet的部分，所以如果剩下的部分差别太大是不行的。
- `resume_from_checkpoint`为`latest`时，实际原理是，对于每1组参数，都会生成1个大体上唯1的特征，然后它会从特征相同的文件夹里选1个step数最大的checkpoint来还原。
- 数据集的格式是这样，图片和同名的 `.txt` prompt文件放在1起就可以了。

```text
dataset/
├── 0001.png
├── 0001.txt
├── 0002.png
├── 0002.txt
└── ...
```

- 如果开启了`use_mask`的话，数据集还要再加上对应的mask文件，比如`0001.png`对应`0001.mask.png`。
  - mask和原本的图片应该是1样的size，然后mask图片白色的像素表示对应的位置要，黑色表示不要。

### 训练与调度
| 参数 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `max_train_steps` | `int` | `10000` | 最大训练总步数 |
| `checkpointing_steps` | `int` | `500` | 保存checkpoint的步数间隔 |
| `validation_steps` | `int` | `100` | 生成验证图的步数间隔 |
| `swap_every_n_steps` | `int` | `8` | 多底模训练时，切换底模的步数间隔 |
| `prior_loss_rate` | `float` | `0.125` | 每步训练先验正则化的概率 |
| `gradient_accumulation_steps` | `int` | `1` | 梯度累积步数 |
| `gradient_checkpointing` | `bool` | `True` | 开启梯度检查点以节省显存 |
| `mixed_precision` | `str` | `None` | 混合精度 |

- swap_every_n_steps不要设得太低，因为交换1次要1.8秒。
- mixed_precision的可选项是`fp16`、`bf16` 或 `None`。


### 优化器
| 参数 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `optimizer` | `str` | `'adam'` | 优化器类型 |
| `lr` | `float` | `1e-4` | 学习率 |
| `lr_scheduler` | `str` | `'constant_with_warmup'` | 调度器类型 |
| `lr_warmup_steps` | `int` | `500` | 学习率预热步数 |
| `adam_beta1` / `adam_beta2` | `float` | `0.9` / `0.999` | Adam的超参数 |
| `adam_weight_decay` | `float` | `0.1` | 权重衰减系数 |
| `max_grad_norm` | `float` | `1.0` | 梯度裁剪阈值 |

- `optimizer`的可选项是`adam`、`8bit_adam`、`prodigy`、`muon`。
- `lr_scheduler`的定义在`diffusers/optimization.py`里，不过建议只选`cosine_with_restarts`或者`constant_with_warmup`。

### LoRA 与损失函数
| 参数 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `rank` | `int` | `32` | 秩 |
| `alpha` | `int` | `rank的1半` | alpha |
| `loss_type` | `str` | `'l2'` | 损失函数 |
| `huber_c` | `float` | `0.1` | Huber损失平滑因子 |
| `snr_gamma` | `float` | `None` | Min-SNR加权系数 |
| `time_min` / `time_max` | `int` | `0` / `1000` | 训练的扩散时间步区间 |

- `loss_type`的可选项是`l2`、`huber`、`huber_scheduled`
- time的0端是纯原图，1000端是纯噪声，如果是训练画风的话，可以考虑设置为`0`到`800`。
  - 先验损失总是不受到时间范围的限制。

### 数据处理
| 参数 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `size_min` / `size_max` | `int` | `704` / `1280` | 随机缩放分辨率范围 |
| `drop_tag_rate` | `float` | `0.0` | 逗号分隔tag的随机丢弃概率 |
| `drop_text_rate` | `float` | `0.0` | 提示词整体随机置空概率 |
| `prompt_post_process` | `str` | `''` | prompt后处理 |
| `prior_loss_prompt_post_process` | `str` | `''` | 先验正则化的prompt后处理 |
| `use_mask` | `bool` | `False` | 是否开启Mask区域加权损失 |
| `mask_min` | `float` | `0.1` | Mask以外非重点区域的最小损失权重 |

- mask_min原本是0，但是发现这样会在背景里产生很多artifact(圣遗物)，所以默认值是0.1。
- prompt后处理的写法是1个Python表达式，比如`s + ', rimochan'`。
  - 原本是设置成了配置函数名+参数的形式，但是试了1下感觉反而很难用，就不过度设计了，这里化繁就简，让大家直接写eval。
  - 输出文件夹里面有`prompt_log.txt`，担心自己的后处理究竟写对了没有的话，可以来看它们。

## 赠品

训练期间的指标是写TensorBoard的，可以这样看: 

```bash
tensorboard --logdir lora_output/logs
```

此外，因为输出是diffusers格式，直接放到SD-WebUI里它会不识别，所以需要用「转换格式.py」转换1下，这个用法就大家自己看吧！


## 结束

就这样，我要去训练群友了，大家88！
