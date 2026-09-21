import time
import torch
import torch.nn as nn



# weight_dict={}
# for layer_id in range(16,24):
#      for expert_id in range(15):
#         # print(layer_id,expert_id)
#         result = torch.load(f'./new_models/layer_{layer_id}/{layer_id}_{expert_id*4}.pth')
#         weight_dict[f'{layer_id}_{expert_id}_up_proj.weight']=result['up_proj.weight']
#         weight_dict[f'{layer_id}_{expert_id}_gate_proj.weight']=result['gate_proj.weight']
#         weight_dict[f'{layer_id}_{expert_id}_down_proj.weight']=result[f'down_proj.weight']
        
# torch.save(weight_dict,'./new_models/w3.pth')
layer_id=2
expert_id=0
weight_dict={}
name='down'
# for i in range(1,4):
result = torch.load(f'./new_models/w{1}.pth')
print(result[f'{layer_id}_{expert_id}_{name}_proj.weight'])


result = torch.load(f'./new_models_old/layer_{layer_id}/{layer_id}_{expert_id}.pth')
print(result[f'{name}_proj.weight'])






