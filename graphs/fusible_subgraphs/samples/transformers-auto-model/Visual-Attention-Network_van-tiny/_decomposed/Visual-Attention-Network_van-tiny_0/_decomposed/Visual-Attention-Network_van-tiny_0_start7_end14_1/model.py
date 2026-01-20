import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, w_0, w_1, w_2, w_3, w_4, w_5, w_6):
        tmp_0 = torch.conv2d(in_2, w_6, w_5, (1, 1), (0, 0), (1, 1), 1)
        tmp_1 = tmp_0 + in_1
        tmp_0 = None
        tmp_2 = w_0.unsqueeze(-1)
        tmp_3 = tmp_2.unsqueeze(-1)
        tmp_2 = None
        tmp_4 = tmp_3 * tmp_1
        tmp_3 = tmp_1 = None
        tmp_5 = in_0 + tmp_4
        tmp_4 = None
        tmp_6 = torch.nn.functional.batch_norm(tmp_5, w_1, w_2, w_4, w_3, False, 0.1, 1e-05)
        return (tmp_5, tmp_6)