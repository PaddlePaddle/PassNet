import torch

class GraphModule(torch.nn.Module):

    def forward(self):
        tmp_0 = torch.zeros((1, 64, 64, 1), dtype=torch.float32)
        return (tmp_0,)