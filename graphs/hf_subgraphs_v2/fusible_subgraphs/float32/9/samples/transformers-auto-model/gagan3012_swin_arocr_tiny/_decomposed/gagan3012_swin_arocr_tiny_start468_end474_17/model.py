import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.nn.functional.layer_norm(in_3, (384,), tmp_1, tmp_0, 1e-05)
        tmp_1 = tmp_0 = None
        tmp_3 = in_2 + tmp_2
        tmp_2 = None
        tmp_4 = tmp_3.view(1, 64, 64, 384)
        tmp_5 = torch.nn.functional.pad(tmp_4, (0, 0, 0, 0, 0, 0), 'constant', None)
        tmp_4 = None
        tmp_6 = tmp_5.view(1, 8, 8, 8, 8, 384)
        tmp_5 = None
        tmp_7 = tmp_6.permute(0, 1, 3, 2, 4, 5)
        tmp_6 = None
        return (tmp_3, tmp_7)