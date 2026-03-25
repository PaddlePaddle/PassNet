import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.conv2d(in_2, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_1 = tmp_0 = None
        tmp_3 = torch.functional.split(tmp_2, split_size_or_sections=[1, 128, 128], dim=1)
        tmp_2 = None
        tmp_4 = tmp_3[0]
        tmp_5 = tmp_3[1]
        tmp_6 = tmp_3[2]
        tmp_3 = None
        return (tmp_5, tmp_4, tmp_6)