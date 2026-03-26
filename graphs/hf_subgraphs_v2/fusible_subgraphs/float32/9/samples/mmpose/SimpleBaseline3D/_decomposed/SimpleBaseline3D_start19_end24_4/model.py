import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = torch.nn.functional.batch_norm(in_5, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 1e-05)
        tmp_0 = tmp_1 = tmp_3 = tmp_2 = None
        tmp_5 = torch.nn.functional.relu(tmp_4, inplace=True)
        tmp_4 = None
        tmp_6 = torch.nn.functional.dropout(tmp_5, 0.5, False, False)
        tmp_5 = None
        tmp_7 = in_4[slice(None, None, None), slice(None, None, None), slice(0, 1, None)]
        tmp_8 = tmp_6 + tmp_7
        tmp_6 = tmp_7 = None
        return (tmp_8,)