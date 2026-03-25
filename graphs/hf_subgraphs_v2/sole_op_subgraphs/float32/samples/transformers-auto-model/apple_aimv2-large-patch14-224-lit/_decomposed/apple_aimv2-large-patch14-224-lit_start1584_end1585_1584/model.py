import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0):
        tmp_0 = w_0
        tmp_1 = tmp_0.clamp(0.0, 4.605170185988092)
        tmp_0 = None
        return (tmp_1,)