import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, w_0, w_1, w_2, w_3, w_4):
        tmp_0 = torch.conv2d(in_1, w_4, None, (1, 1), (1, 1), (1, 1), 1)
        tmp_1 = torch.nn.functional.batch_norm(tmp_0, w_0, w_1, w_3, w_2, False, 0.1, 1e-05)
        tmp_0 = None
        tmp_1 += in_0
        tmp_2 = tmp_1
        tmp_1 = None
        tmp_3 = torch.nn.functional.relu(tmp_2, inplace=True)
        tmp_2 = None
        return (tmp_3,)