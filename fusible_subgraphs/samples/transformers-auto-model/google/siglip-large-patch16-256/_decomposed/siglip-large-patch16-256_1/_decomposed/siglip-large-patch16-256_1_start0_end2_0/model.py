import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.reshape(2, 64, 1024)
        tmp_1 = tmp_0.contiguous()
        tmp_0 = None
        return (tmp_1,)