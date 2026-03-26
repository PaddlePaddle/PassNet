import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_0
        tmp_1 = torch.nn.functional.relu(in_1, inplace=True)
        tmp_2 = torch.flatten(tmp_1, 2)
        tmp_1 = None
        tmp_3 = torch.functional.norm(tmp_2, dim=-1, keepdim=True)
        tmp_4 = tmp_3 * 0.07216878364870322
        tmp_3 = None
        tmp_5 = tmp_4.clamp(min=1e-05)
        tmp_4 = None
        tmp_6 = tmp_2 / tmp_5
        tmp_2 = tmp_5 = None
        tmp_7 = tmp_6 * tmp_0
        tmp_6 = tmp_0 = None
        return (tmp_7,)