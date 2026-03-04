import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = -in_3
        tmp_1 = torch.cat((tmp_0, in_2), dim=-1)
        tmp_0 = None
        tmp_2 = tmp_1 * in_0
        tmp_1 = None
        tmp_3 = in_1 + tmp_2
        tmp_2 = None
        tmp_4 = tmp_3.to(dtype=torch.float32)
        tmp_3 = None
        tmp_5 = tmp_4.transpose(-1, -2)
        tmp_4 = None
        return (tmp_5,)