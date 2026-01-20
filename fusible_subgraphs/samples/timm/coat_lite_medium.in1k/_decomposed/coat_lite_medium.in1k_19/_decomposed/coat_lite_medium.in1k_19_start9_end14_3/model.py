import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1, w_2, w_3, w_4):
        tmp_0 = torch.conv2d(in_0, w_4, w_3, (2, 2), (0, 0), (1, 1), 1)
        tmp_1 = tmp_0.flatten(2)
        tmp_0 = None
        tmp_2 = tmp_1.transpose(1, 2)
        tmp_1 = None
        tmp_3 = torch.nn.functional.layer_norm(tmp_2, (512,), w_1, w_0, 1e-05)
        tmp_2 = None
        tmp_4 = w_2.expand(1, -1, -1)
        return (tmp_3, tmp_4)