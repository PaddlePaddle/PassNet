import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = torch.nn.functional.silu(in_5, inplace=True)
        tmp_6 = torch.conv2d(tmp_5, tmp_4, None, (1, 1), (1, 1), (1, 1), 1)
        tmp_5 = tmp_4 = None
        tmp_7 = torch.nn.functional.avg_pool2d(tmp_6, 2, 2, 0, False, True, None)
        tmp_8 = torch.nn.functional.batch_norm(tmp_6, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 1e-05)
        tmp_6 = tmp_0 = tmp_1 = tmp_3 = tmp_2 = None
        tmp_9 = torch.nn.functional.silu(tmp_8, inplace=True)
        tmp_8 = None
        tmp_10 = torch.nn.functional.avg_pool2d(tmp_9, 2, 2, 0, False, True, None)
        tmp_9 = None
        return (tmp_7, tmp_10)