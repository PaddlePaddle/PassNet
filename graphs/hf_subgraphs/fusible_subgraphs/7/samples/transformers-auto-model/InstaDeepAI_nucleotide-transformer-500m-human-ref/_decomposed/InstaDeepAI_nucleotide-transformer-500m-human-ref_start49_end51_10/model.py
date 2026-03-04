import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.contiguous()
        tmp_1 = tmp_0.view((64, 128, 1280))
        tmp_0 = None
        return (tmp_1,)