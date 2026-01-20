import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1):
        tmp_0 = torch.nn.functional.group_norm(in_0, 512, w_1, w_0, 1e-05)
        tmp_1 = torch.nn.functional.gelu(tmp_0)
        tmp_0 = None
        return (tmp_1,)