import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2):
        tmp_1 = torch.nn.functional.relu(in_2, inplace = True);  in_2 = None
        tmp_2 = tmp_1.mean((2, 3))
        tmp_3 = tmp_2.view(1, 1, -1);  tmp_2 = None
        to = tmp_3.to(torch.float16);  tmp_3 = None
        conv1d = torch.conv1d(to, in_0, None, (1,), (2,), (1,), 1);  to = in_0 = None
        tmp_5 = conv1d.sigmoid();  conv1d = None
        tmp_6 = tmp_5.view(1, -1, 1, 1);  tmp_5 = None
        tmp_7 = tmp_6.expand_as(tmp_1);  tmp_6 = None
        tmp_8 = tmp_1 * tmp_7;  tmp_1 = tmp_7 = None
        tmp_9 = tmp_8 + in_1;  tmp_8 = in_1 = None
        tmp_10 = torch.nn.functional.max_pool2d(tmp_9, 3, 2, 0, 1, ceil_mode = True, return_indices = False);  tmp_9 = None
        return (tmp_10,)
        