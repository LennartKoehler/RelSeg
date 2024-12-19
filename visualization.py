import torch
import torchviz

def visualize(model, model_shape):
    x = torch.randn(model_shape, dtype=torch.float).to("cuda")
    y = model(x)
    print(y.shape)
    dot = torchviz.make_dot(y, show_saved=True)
    dot.format = 'png'
    dot.render('visualization/model_arch')