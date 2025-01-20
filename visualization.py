import torch
import torchviz

def visualize(model, model_shape):
    x = torch.randn(model_shape, dtype=torch.float).to("cuda")
    y = model(x)
    print(y.shape)
    dot = torchviz.make_dot(y, params=dict(model.named_parameters()), show_saved=True)
    dot.format = 'pdf'
    dot.render('visualization/model_arch_sup')