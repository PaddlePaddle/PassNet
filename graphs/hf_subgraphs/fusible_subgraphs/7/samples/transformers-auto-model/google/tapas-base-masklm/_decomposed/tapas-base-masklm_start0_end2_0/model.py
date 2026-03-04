import torch

class GraphModule(torch.nn.Module):

    def forward(self):
        tmp_0 = torch.tensor([1])
        tmp_1 = torch.prod(tmp_0)
        tmp_0 = None
        return (tmp_1,)