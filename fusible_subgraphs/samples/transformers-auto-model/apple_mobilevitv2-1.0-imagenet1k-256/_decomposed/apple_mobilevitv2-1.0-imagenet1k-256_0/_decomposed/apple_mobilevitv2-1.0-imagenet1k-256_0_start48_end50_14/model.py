import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1):
        tmp_0 = torch.nn.functional.group_norm(in_0, 1, w_1, w_0, 1e-05)
        tmp_1 = tmp_0.reshape(1, 512, 256)
        tmp_0 = None
        return (tmp_1,)