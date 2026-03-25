import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0):
        tmp_0 = w_0
        tmp_1 = tmp_0 * 0.34412564994580647
        tmp_0 = None
        return (tmp_1,)