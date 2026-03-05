import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.reshape(in_0, (1, 576, 768))
        return (tmp_0,)