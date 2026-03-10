import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0):
        tmp_0 = w_0
        tmp_1 = torch.clamp(tmp_0, max=4.605170185988092)
        tmp_0 = None
        return (tmp_1,)