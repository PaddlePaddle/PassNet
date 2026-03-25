import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3):
        tmp_2 = torch.nn.functional.relu(in_3, inplace = True);  in_3 = None
        tmp_3 = tmp_2.mean((2, 3), keepdim = True)
        conv2d = torch.conv2d(tmp_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1);  tmp_3 = in_1 = in_0 = None
        tmp_5 = torch.nn.functional.hardsigmoid(conv2d, False);  conv2d = None
        tmp_6 = tmp_2 * tmp_5;  tmp_2 = tmp_5 = None
        tmp_7 = tmp_6 + in_2;  tmp_6 = in_2 = None
        tmp_8 = torch.nn.functional.max_pool2d(tmp_7, 3, 2, 0, 1, ceil_mode = True, return_indices = False);  tmp_7 = None
        return (tmp_8,)
        