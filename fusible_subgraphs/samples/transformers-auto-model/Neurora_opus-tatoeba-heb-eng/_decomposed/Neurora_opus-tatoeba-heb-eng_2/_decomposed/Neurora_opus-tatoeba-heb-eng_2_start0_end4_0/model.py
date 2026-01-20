import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, w_0, w_1, w_2, w_3):
        tmp_0 = torch.nn.functional.linear(in_0, w_3, w_2)
        tmp_1 = torch.nn.functional.dropout(tmp_0, p=0.1, training=False)
        tmp_0 = None
        tmp_2 = in_1 + tmp_1
        tmp_1 = None
        tmp_3 = torch.nn.functional.layer_norm(tmp_2, (512,), w_1, w_0, 1e-05)
        tmp_2 = None
        return (tmp_3,)