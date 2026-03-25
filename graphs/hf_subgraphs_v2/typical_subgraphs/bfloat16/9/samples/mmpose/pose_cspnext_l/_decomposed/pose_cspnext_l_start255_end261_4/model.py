import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, in_0, in_1):
        tmp_2 = torch.nn.functional.silu(in_1, inplace = True);  in_1 = None
        tmp_3 = torch.cat((tmp_2, in_0), dim = 1);  tmp_2 = in_0 = None
        tmp_4 = torch.nn.functional.adaptive_avg_pool2d(tmp_3, 1)
        conv2d = torch.conv2d(tmp_4, w_1, w_0, (1, 1), (0, 0), (1, 1), 1);  tmp_4 = w_1 = w_0 = None
        tmp_6 = torch.nn.functional.hardsigmoid(conv2d, True);  conv2d = None
        tmp_7 = tmp_3 * tmp_6;  tmp_3 = tmp_6 = None
        return (tmp_7,)
        