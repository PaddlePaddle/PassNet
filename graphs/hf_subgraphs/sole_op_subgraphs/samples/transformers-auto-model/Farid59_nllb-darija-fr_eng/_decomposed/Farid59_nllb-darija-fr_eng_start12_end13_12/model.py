import torch

class GraphModule(torch.nn.Module):

    def forward(self):
        tmp_0 = torch.tensor(1.0, dtype=torch.bfloat16)
        return (tmp_0,)