import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5):
        tmp_0 = torch.nn.functional.silu(in_0, inplace=True)
        tmp_1 = torch.nn.functional.interpolate(in_1, size=(320, 320), mode='nearest')
        tmp_2 = torch.nn.functional.interpolate(in_2, size=(320, 320), mode='nearest')
        tmp_3 = torch.nn.functional.interpolate(in_3, size=(320, 320), mode='nearest')
        tmp_4 = torch.nn.functional.interpolate(in_4, size=(320, 320), mode='nearest')
        tmp_5 = torch.nn.functional.interpolate(in_5, size=(320, 320), mode='nearest')
        tmp_6 = torch.stack([tmp_1, tmp_2, tmp_3, tmp_4, tmp_5, tmp_0])
        tmp_1 = tmp_2 = tmp_3 = tmp_4 = tmp_5 = tmp_0 = None
        tmp_7 = torch.sum(tmp_6, dim=0)
        tmp_6 = None
        return (tmp_7,)