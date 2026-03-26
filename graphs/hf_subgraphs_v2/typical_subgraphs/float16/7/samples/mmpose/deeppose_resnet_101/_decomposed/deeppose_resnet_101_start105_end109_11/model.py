import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6):
        tmp_5 = torch.nn.functional.relu(in_6, inplace = True);  in_6 = None
        conv2d = torch.conv2d(tmp_5, in_4, None, (1, 1), (0, 0), (1, 1), 1);  tmp_5 = in_4 = None
        tmp_7 = torch.nn.functional.batch_norm(conv2d, in_0, in_1, in_3, in_2, False, 0.1, 1e-05);  conv2d = in_0 = in_1 = in_3 = in_2 = None
        tmp_7 += in_5;  tmp_8 = tmp_7;  tmp_7 = in_5 = None
        return (tmp_8,)
        