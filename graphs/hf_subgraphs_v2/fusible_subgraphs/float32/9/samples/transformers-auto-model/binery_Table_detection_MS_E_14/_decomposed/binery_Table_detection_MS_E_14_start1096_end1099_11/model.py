import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4 + in_5
        tmp_5 = torch.nn.functional.layer_norm(tmp_4, (256,), tmp_3, tmp_2, 1e-05)
        tmp_4 = tmp_3 = tmp_2 = None
        tmp_6 = torch.nn.functional.layer_norm(tmp_5, (256,), tmp_1, tmp_0, 1e-05)
        tmp_5 = tmp_1 = tmp_0 = None
        return (tmp_6,)