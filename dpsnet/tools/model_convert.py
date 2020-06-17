import torch
model = torch.load("/data/jianzh/20200418_rpc_render_final_finetune_1_v1/model_0001000.pth")
print(model.keys())
# Remove the previous training parameters. 
del model['iteration']
torch.save(model, "/data/jianzh/20200420_rpc_render_final_finetune_1_v2/best_model.pth")