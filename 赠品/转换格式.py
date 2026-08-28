import torch
from safetensors.torch import load_file, save_file
from diffusers.utils import convert_all_state_dict_to_peft, convert_state_dict_to_kohya


def 转换格式(input_lora, output_lora, alpha, metadata={}):
    diffusers_state_dict = load_file(input_lora)
    peft_state_dict = convert_all_state_dict_to_peft(diffusers_state_dict)
    kohya_state_dict = convert_state_dict_to_kohya(peft_state_dict)
    for k, v in [*kohya_state_dict.items()]:
        if k.endswith('.alpha'):
            kohya_state_dict[k] = torch.tensor(float(alpha))
        else:
            kohya_state_dict[k] = v.to('cuda').to(torch.float16).to('cpu')
    save_file(kohya_state_dict, output_lora, metadata={'source': str(input_lora), 'ss_base_model_version': 'sdxl_base_v1-0', 'ss_network_module': 'networks.lora', 'repo': 'rimo_lora_lab'} | {k: str(v) for k, v in metadata.items()})


# metadata = json.load(open(f'{名字}/metadata.json', encoding='utf8'))
# alpha = metadata['alpha']
# convert_and_save(f'{名字}/checkpoint-6000/pytorch_lora_weights.safetensors', f'R:/stable-diffusion-webui-master/models/Lora/my_lora.safetensors', alpha, metadata)
