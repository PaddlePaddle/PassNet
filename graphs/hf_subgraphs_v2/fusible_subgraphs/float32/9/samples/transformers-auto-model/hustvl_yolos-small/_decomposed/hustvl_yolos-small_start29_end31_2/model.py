import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.cat((in_2, in_5, in_3), dim=2)
        tmp_3 = torch.nn.functional.layer_norm(in_4, (384,), tmp_1, tmp_0, 1e-12)
        tmp_1 = tmp_0 = None
        return (tmp_3, tmp_2)