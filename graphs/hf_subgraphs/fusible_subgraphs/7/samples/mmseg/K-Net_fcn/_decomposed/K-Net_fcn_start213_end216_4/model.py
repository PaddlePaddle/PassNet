import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.nn.functional.layer_norm(in_3, (256,), tmp_1, tmp_0, 1e-05)
        tmp_1 = tmp_0 = None
        tmp_3 = in_2.sigmoid()
        tmp_4 = tmp_2.sigmoid()
        tmp_2 = None
        return (tmp_3, tmp_4)