import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1, w_2, w_3):
        tmp_0 = torch.nn.functional.batch_norm(in_0, w_0, w_1, w_3, w_2, False, 0.1, 1e-05)
        tmp_1 = torch.nn.functional.relu(tmp_0, inplace=False)
        tmp_0 = None
        return (tmp_1,)