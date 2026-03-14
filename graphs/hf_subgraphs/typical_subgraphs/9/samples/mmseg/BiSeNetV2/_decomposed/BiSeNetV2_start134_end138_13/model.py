import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, in_0, in_1):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = in_1 + in_0
        tmp_5 = torch.nn.functional.relu(tmp_4, inplace=False)
        tmp_4 = None
        tmp_6 = torch.nn.functional.adaptive_avg_pool2d(tmp_5, (1, 1))
        tmp_7 = torch.nn.functional.batch_norm(tmp_6, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 1e-05)
        tmp_6 = tmp_0 = tmp_1 = tmp_3 = tmp_2 = None
        return (tmp_7, tmp_5)