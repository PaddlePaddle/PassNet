import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = in_0
        tmp_1 = torch.nn.functional.linear(in_2, tmp_0, None)
        tmp_0 = None
        tmp_2 = tmp_1.view((128, 64, -1, 128))
        tmp_1 = None
        tmp_3 = tmp_2.transpose(1, 2)
        tmp_2 = None
        tmp_4 = in_1.unsqueeze(1)
        tmp_5 = in_3.unsqueeze(1)
        return (tmp_4, tmp_5, tmp_3)