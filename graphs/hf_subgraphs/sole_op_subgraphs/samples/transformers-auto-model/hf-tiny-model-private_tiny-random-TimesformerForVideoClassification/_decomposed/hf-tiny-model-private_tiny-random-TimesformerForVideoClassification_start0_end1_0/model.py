import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0
        tmp_1 = tmp_0.reshape(2, 3, 10, 10)
        tmp_0 = None
        return (tmp_1,)