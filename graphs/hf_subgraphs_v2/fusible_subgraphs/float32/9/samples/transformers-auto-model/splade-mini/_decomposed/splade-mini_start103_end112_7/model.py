import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = in_6 + in_5
        tmp_6 = torch.nn.functional.layer_norm(tmp_5, (256,), tmp_2, tmp_1, 1e-12)
        tmp_5 = tmp_2 = tmp_1 = None
        tmp_7 = tmp_6[slice(None, None, None), 0]
        tmp_8 = torch.nn.functional.linear(tmp_7, tmp_4, tmp_3)
        tmp_7 = tmp_4 = tmp_3 = None
        tmp_9 = torch.tanh(tmp_8)
        tmp_8 = tmp_9 = None
        tmp_10 = tmp_0.unsqueeze(-1)
        tmp_0 = None
        tmp_11 = tmp_10.expand((1, 10, 256))
        tmp_10 = None
        tmp_12 = tmp_11.to(torch.float32)
        tmp_11 = None
        tmp_13 = tmp_6 * tmp_12
        return (tmp_6, tmp_12, tmp_13)