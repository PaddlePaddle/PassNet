import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.nn.functional.softmax(in_0, dim=-1, dtype=torch.float32)
        return (tmp_0,)