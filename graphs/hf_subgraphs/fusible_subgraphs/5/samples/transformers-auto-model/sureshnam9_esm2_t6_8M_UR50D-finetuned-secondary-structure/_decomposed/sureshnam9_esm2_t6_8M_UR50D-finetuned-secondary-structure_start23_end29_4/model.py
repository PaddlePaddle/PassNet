import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_4 / in_3
        tmp_4 = tmp_3.to(torch.float32)
        tmp_3 = None
        tmp_5 = tmp_0.unsqueeze(-1)
        tmp_0 = None
        tmp_6 = tmp_4 * tmp_5
        tmp_4 = tmp_5 = None
        tmp_7 = tmp_6.to(torch.float32)
        tmp_6 = None
        tmp_8 = torch.nn.functional.layer_norm(tmp_7, (320,), tmp_2, tmp_1, 1e-05)
        tmp_2 = tmp_1 = None
        return (tmp_7, tmp_8)