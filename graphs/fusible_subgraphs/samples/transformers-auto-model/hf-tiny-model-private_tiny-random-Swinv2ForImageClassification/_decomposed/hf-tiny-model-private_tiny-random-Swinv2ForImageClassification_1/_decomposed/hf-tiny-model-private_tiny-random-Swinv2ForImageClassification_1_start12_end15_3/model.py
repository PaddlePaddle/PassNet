import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1):
        tmp_0 = torch.nn.functional.linear(in_0, w_1, None)
        tmp_1 = tmp_0.view(-1, 4)
        tmp_0 = None
        tmp_2 = w_0.view(-1)
        return (tmp_1, tmp_2)