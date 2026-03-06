import torch

class GraphModule(torch.nn.Module):

    def forward(self):
        tmp_0 = torch.zeros((1, 256, 256, 1), dtype=torch.float32)
        return (tmp_0,)