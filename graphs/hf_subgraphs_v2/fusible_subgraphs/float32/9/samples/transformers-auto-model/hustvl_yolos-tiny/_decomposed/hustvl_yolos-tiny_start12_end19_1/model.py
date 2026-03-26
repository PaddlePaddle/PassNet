import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.nn.functional.interpolate(in_5, size=(32, 32), mode='bicubic', align_corners=False)
        tmp_3 = tmp_2.flatten(2)
        tmp_2 = None
        tmp_4 = tmp_3.transpose(1, 2)
        tmp_3 = None
        tmp_5 = torch.cat((in_2, tmp_4, in_3), dim=1)
        tmp_4 = None
        tmp_6 = in_4 + tmp_5
        tmp_5 = None
        tmp_7 = torch.nn.functional.dropout(tmp_6, 0.0, False, False)
        tmp_6 = None
        tmp_8 = torch.nn.functional.layer_norm(tmp_7, (192,), tmp_1, tmp_0, 1e-12)
        tmp_1 = tmp_0 = None
        return (tmp_7, tmp_8)