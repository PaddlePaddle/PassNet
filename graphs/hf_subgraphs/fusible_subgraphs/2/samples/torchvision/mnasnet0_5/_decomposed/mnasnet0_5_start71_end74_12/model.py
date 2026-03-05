import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = torch.conv2d(in_5, tmp_0, None, (1, 1), (2, 2), (1, 1), 240)
        tmp_0 = None
        tmp_6 = torch.nn.functional.batch_norm(tmp_5, tmp_1, tmp_2, tmp_4, tmp_3, False, 0.00029999999999996696, 1e-05)
        tmp_5 = tmp_1 = tmp_2 = tmp_4 = tmp_3 = None
        tmp_7 = torch.nn.functional.relu(tmp_6, inplace=True)
        tmp_6 = None
        return (tmp_7,)