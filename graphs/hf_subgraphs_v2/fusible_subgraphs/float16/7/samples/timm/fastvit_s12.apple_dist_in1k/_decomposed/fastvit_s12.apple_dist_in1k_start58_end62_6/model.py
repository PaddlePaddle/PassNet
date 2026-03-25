import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4):
        conv2d = torch.conv2d(in_4, in_2, in_1, (1, 1), (0, 0), (1, 1), 1);  in_4 = in_2 = in_1 = None
        tmp_4 = torch.nn.functional.dropout(conv2d, 0.0, False, False);  conv2d = None
        tmp_5 = tmp_4 * in_0;  tmp_4 = in_0 = None
        tmp_6 = in_3 + tmp_5;  in_3 = tmp_5 = None
        return (tmp_6,)
        