import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.nn.functional.layer_norm(in_2, (256,), tmp_1, tmp_0, 1e-05)
        tmp_1 = tmp_0 = None
        tmp_3 = torch.nn.functional.relu(tmp_2, inplace=True)
        tmp_2 = None
        tmp_4 = tmp_3.reshape(1, 150, -1)
        tmp_3 = None
        tmp_5 = tmp_4.permute(1, 0, 2)
        tmp_4 = None
        return (tmp_5,)