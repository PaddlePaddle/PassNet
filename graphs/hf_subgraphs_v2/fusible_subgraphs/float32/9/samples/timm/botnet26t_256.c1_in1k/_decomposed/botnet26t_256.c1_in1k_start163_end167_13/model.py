import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4.reshape(1, 512, 16, 16)
        tmp_5 = torch.nn.functional.avg_pool2d(tmp_4, 2, 2, 0, False, True, None)
        tmp_4 = None
        tmp_6 = torch.nn.functional.batch_norm(tmp_5, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 1e-05)
        tmp_5 = tmp_0 = tmp_1 = tmp_3 = tmp_2 = None
        tmp_7 = torch.nn.functional.relu(tmp_6, inplace=True)
        tmp_6 = None
        return (tmp_7,)