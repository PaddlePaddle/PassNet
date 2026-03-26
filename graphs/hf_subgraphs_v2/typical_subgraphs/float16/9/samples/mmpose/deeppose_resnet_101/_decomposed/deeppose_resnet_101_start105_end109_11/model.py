import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, in_0, in_1):
        tmp_5 = torch.nn.functional.relu(in_1, inplace = True);  in_1 = None
        conv2d = torch.conv2d(tmp_5, w_4, None, (1, 1), (0, 0), (1, 1), 1);  tmp_5 = w_4 = None
        tmp_7 = torch.nn.functional.batch_norm(conv2d, w_0, w_1, w_3, w_2, False, 0.1, 1e-05);  conv2d = w_0 = w_1 = w_3 = w_2 = None
        tmp_7 += in_0;  tmp_8 = tmp_7;  tmp_7 = in_0 = None
        return (tmp_8,)
        