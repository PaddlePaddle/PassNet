import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0
        tmp_1 = tmp_0.unsqueeze(1)
        tmp_0 = None
        tmp_2 = tmp_1.transpose(2, 3)
        tmp_1 = None
        return (tmp_2,)