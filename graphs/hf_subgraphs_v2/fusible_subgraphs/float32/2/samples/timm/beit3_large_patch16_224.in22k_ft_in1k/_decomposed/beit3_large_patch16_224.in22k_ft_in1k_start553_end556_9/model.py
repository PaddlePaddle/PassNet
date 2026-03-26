import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.nn.functional.gelu(in_2, approximate='none')
        tmp_3 = torch.nn.functional.dropout(tmp_2, 0.0, False, False)
        tmp_2 = None
        tmp_4 = torch.nn.functional.layer_norm(tmp_3, (4096,), tmp_1, tmp_0, 1e-05)
        tmp_3 = tmp_1 = tmp_0 = None
        return (tmp_4,)