import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = torch.functional.meshgrid(in_1, in_0, indexing='xy')
        return (tmp_0,)