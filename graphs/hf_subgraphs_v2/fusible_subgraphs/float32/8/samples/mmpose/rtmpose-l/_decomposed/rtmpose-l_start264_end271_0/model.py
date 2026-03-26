import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = torch.conv2d(in_3, tmp_1, tmp_0, (1, 1), (3, 3), (1, 1), 1)
        tmp_1 = tmp_0 = None
        tmp_4 = torch.flatten(tmp_3, 2)
        tmp_3 = None
        tmp_5 = torch.functional.norm(tmp_4, dim=-1, keepdim=True)
        tmp_6 = tmp_5 * 0.14433756729740643
        tmp_5 = None
        tmp_7 = tmp_6.clamp(min=1e-05)
        tmp_6 = None
        tmp_8 = tmp_4 / tmp_7
        tmp_4 = tmp_7 = None
        tmp_9 = tmp_8 * tmp_2
        tmp_8 = tmp_2 = None
        return (tmp_9,)