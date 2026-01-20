import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1, w_2, w_3):
        tmp_0 = in_0.reshape(1, 512, 16, 16)
        tmp_1 = torch.nn.functional.avg_pool2d(tmp_0, 2, 2, 0, False, True, None)
        tmp_0 = None
        tmp_2 = torch.nn.functional.batch_norm(tmp_1, w_0, w_1, w_3, w_2, False, 0.1, 1e-05)
        tmp_1 = None
        tmp_3 = torch.nn.functional.relu(tmp_2, inplace=True)
        tmp_2 = None
        return (tmp_3,)