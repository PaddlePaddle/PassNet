import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, w_0, w_1, w_2, w_3):
        tmp_0 = torch.conv2d(in_1, w_1, w_0, (1, 1), (1, 1), (1, 1), 256)
        tmp_1 = tmp_0 + in_1
        tmp_0 = None
        tmp_2 = tmp_1.flatten(2)
        tmp_1 = None
        tmp_3 = tmp_2.transpose(1, 2)
        tmp_2 = None
        tmp_4 = torch.cat((in_0, tmp_3), dim=1)
        tmp_3 = None
        tmp_5 = torch.nn.functional.layer_norm(tmp_4, (256,), w_3, w_2, 1e-06)
        return (tmp_4, tmp_5)