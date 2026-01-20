import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1, w_2, w_3):
        tmp_0 = torch.conv2d(in_0, w_3, w_2, (4, 4), (0, 0), (1, 1), 1)
        tmp_1 = tmp_0.reshape(1, 128, -1)
        tmp_0 = None
        tmp_2 = tmp_1.permute(0, 2, 1)
        tmp_1 = None
        tmp_3 = torch.nn.functional.layer_norm(tmp_2, (128,), w_1, w_0, 1e-05)
        tmp_2 = None
        return (tmp_3,)