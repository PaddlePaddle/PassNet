import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, in_0, in_1):
        tmp_5 = torch.nn.functional.relu(in_1, inplace = True);  in_1 = None
        conv2d = torch.conv2d(tmp_5, w_0, None, (1, 1), (1, 1), (1, 1), 1);  tmp_5 = w_0 = None
        tmp_7 = conv2d + in_0;  conv2d = in_0 = None
        tmp_8 = torch.nn.functional.batch_norm(tmp_7, w_1, w_2, w_4, w_3, False, 0.1, 1e-05);  w_1 = w_2 = w_4 = w_3 = None
        tmp_9 = torch.nn.functional.relu(tmp_8, inplace = True);  tmp_8 = None
        return (tmp_7, tmp_9)
        