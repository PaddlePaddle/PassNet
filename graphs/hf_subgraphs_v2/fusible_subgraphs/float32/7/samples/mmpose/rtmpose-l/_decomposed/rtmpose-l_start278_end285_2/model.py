import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_0
        tmp_1 = torch.nn.functional.silu(in_1, inplace=True)
        tmp_2 = torch.functional.split(tmp_1, [512, 512, 128], dim=2)
        tmp_1 = None
        tmp_3 = tmp_2[0]
        tmp_4 = tmp_2[1]
        tmp_5 = tmp_2[2]
        tmp_2 = None
        tmp_6 = tmp_5.unsqueeze(2)
        tmp_5 = None
        tmp_7 = tmp_0[None, None, slice(None, None, None)]
        tmp_0 = None
        return (tmp_7, tmp_3, tmp_6, tmp_4)