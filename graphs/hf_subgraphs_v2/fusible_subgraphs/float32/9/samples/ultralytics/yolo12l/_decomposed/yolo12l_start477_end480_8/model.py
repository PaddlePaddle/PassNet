import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.reshape(1, 20, 20, 256)
        tmp_1 = tmp_0.permute(0, 3, 1, 2)
        tmp_0 = None
        tmp_2 = tmp_1.contiguous()
        tmp_1 = None
        return (tmp_2,)