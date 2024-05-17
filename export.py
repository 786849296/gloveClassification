import torch

model = torch.load("GloveNet_k0.pt").cpu()
dummy_input = torch.randn(16, 9, 10, 16, requires_grad=True)
torch.onnx.export(model, dummy_input, "GloveNet.onnx", opset_version=15, verbose=True, 
    input_names = ['input'],   # the model's input names 
    output_names = ['output'], # the model's output names 
    dynamic_axes={
        'input' : {0 : 'batch_size'},    # variable length axes 
        'output' : {0 : 'batch_size'}
})
