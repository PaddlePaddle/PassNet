import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0):
        tmp_0 = w_0
        tmp_1 = tmp_0 * 0.02551551815399144
        tmp_0 = None
        return (tmp_1,)