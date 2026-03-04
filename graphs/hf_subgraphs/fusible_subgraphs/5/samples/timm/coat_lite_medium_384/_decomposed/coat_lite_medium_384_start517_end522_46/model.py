import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = torch.cat([in_0, in_1, in_2], dim=1)
        tmp_1 = tmp_0.reshape(1, 8, 40, 576)
        tmp_0 = None
        tmp_2 = tmp_1.transpose(-1, -2)
        tmp_1 = None
        tmp_3 = in_3 * tmp_2
        tmp_2 = None
        tmp_4 = torch.nn.functional.pad(tmp_3, (0, 0, 1, 0, 0, 0), 'constant', None)
        tmp_3 = None
        return (tmp_4,)