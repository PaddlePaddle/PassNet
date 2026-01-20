import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, w_0, w_1, w_2, w_3, w_4, w_5):
        tmp_0 = torch.conv2d(in_1, w_1, w_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_1 = tmp_0.sigmoid()
        tmp_0 = None
        tmp_2 = in_0 * tmp_1
        tmp_1 = None
        tmp_3 = torch.nn.functional.batch_norm(tmp_2, w_2, w_3, w_5, w_4, False, 0.1, 1e-05)
        tmp_2 = None
        tmp_4 = torch.nn.functional.silu(tmp_3, inplace=True)
        tmp_3 = None
        return (tmp_4,)