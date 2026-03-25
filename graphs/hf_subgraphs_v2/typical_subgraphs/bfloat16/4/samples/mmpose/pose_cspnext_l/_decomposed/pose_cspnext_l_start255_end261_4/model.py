import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3):
        tmp_2 = torch.nn.functional.silu(in_3, inplace = True);  in_3 = None
        tmp_3 = torch.cat((tmp_2, in_2), dim = 1);  tmp_2 = in_2 = None
        tmp_4 = torch.nn.functional.adaptive_avg_pool2d(tmp_3, 1)
        conv2d = torch.conv2d(tmp_4, in_1, in_0, (1, 1), (0, 0), (1, 1), 1);  tmp_4 = in_1 = in_0 = None
        tmp_6 = torch.nn.functional.hardsigmoid(conv2d, True);  conv2d = None
        tmp_7 = tmp_3 * tmp_6;  tmp_3 = tmp_6 = None
        return (tmp_7,)
        