import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0):
        tmp_0 = w_0
        tmp_1 = torch.zeros_like(tmp_0, requires_grad=False)
        tmp_0 = None
        return (tmp_1,)