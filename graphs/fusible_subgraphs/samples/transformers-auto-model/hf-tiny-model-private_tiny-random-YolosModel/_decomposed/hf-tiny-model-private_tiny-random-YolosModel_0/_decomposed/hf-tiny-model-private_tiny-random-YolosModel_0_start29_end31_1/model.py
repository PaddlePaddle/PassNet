import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, w_0, w_1):
        tmp_0 = torch.cat((in_1, in_3, in_2), dim=2)
        tmp_1 = torch.nn.functional.layer_norm(in_0, (32,), w_1, w_0, 1e-12)
        return (tmp_0, tmp_1)