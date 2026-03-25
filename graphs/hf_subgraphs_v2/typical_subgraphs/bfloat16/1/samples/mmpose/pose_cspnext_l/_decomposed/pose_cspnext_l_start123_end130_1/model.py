import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4):
        tmp_2 = torch.nn.functional.silu(in_3, inplace = True);  in_3 = None
        tmp_3 = tmp_2 + in_2;  tmp_2 = in_2 = None
        tmp_4 = torch.cat((tmp_3, in_4), dim = 1);  tmp_3 = in_4 = None
        tmp_5 = torch.nn.functional.adaptive_avg_pool2d(tmp_4, 1)
        conv2d = torch.conv2d(tmp_5, in_1, in_0, (1, 1), (0, 0), (1, 1), 1);  tmp_5 = in_1 = in_0 = None
        tmp_7 = torch.nn.functional.hardsigmoid(conv2d, True);  conv2d = None
        tmp_8 = tmp_4 * tmp_7;  tmp_4 = tmp_7 = None
        return (tmp_8,)
        