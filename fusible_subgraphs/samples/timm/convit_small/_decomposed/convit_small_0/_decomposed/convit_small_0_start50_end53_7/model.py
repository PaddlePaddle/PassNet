import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0):
        tmp_0 = torch.nn.functional.linear(in_0, w_0, None)
        tmp_1 = tmp_0.reshape(1, 196, 9, 48)
        tmp_0 = None
        tmp_2 = tmp_1.permute(0, 2, 1, 3)
        tmp_1 = None
        return (tmp_2,)