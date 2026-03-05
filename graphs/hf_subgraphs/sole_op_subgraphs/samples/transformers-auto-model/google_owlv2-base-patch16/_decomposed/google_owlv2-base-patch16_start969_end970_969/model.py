import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.reshape(in_0, (1, 3600, 768))
        return (tmp_0,)