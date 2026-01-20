import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1):
        tmp_0 = torch.conv2d(in_0, w_1, w_0, (1, 1), (1, 1), (1, 1), 2048)
        tmp_1 = tmp_0.flatten(2)
        tmp_0 = None
        tmp_2 = tmp_1.transpose(1, 2)
        tmp_1 = None
        tmp_3 = torch.nn.functional.gelu(tmp_2)
        tmp_2 = None
        tmp_4 = torch.nn.functional.dropout(tmp_3, 0.0, False, False)
        tmp_3 = None
        return (tmp_4,)