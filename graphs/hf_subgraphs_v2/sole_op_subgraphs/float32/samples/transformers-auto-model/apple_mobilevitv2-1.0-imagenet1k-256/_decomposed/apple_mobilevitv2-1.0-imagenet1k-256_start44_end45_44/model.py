import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.functional.split(in_0, split_size_or_sections=[1, 128, 128], dim=1)
        return (tmp_0,)