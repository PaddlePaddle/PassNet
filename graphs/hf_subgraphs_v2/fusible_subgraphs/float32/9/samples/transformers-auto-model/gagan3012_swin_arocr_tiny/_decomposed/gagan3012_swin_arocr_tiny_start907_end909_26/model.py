import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.contiguous()
        tmp_1 = tmp_0.view((16, 64, 768))
        tmp_0 = None
        return (tmp_1,)