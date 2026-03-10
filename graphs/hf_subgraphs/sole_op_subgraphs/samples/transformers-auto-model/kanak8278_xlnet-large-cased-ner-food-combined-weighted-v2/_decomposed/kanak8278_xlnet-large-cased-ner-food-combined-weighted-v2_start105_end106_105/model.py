import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.nn.functional.softmax(in_0, dim=3)
        return (tmp_0,)