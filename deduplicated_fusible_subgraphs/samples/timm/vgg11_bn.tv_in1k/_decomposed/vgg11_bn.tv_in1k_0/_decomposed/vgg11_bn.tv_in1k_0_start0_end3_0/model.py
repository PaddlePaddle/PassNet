import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1, w_2, w_3, w_4, w_5):
        tmp_0 = torch.conv2d(in_0, w_1, w_0, (1, 1), (1, 1), (1, 1), 1)
        tmp_1 = torch.nn.functional.batch_norm(tmp_0, w_2, w_3, w_5, w_4, False, 0.1, 1e-05)
        tmp_0 = None
        tmp_2 = torch.nn.functional.relu(tmp_1, inplace=True)
        tmp_1 = None
        return (tmp_2,)