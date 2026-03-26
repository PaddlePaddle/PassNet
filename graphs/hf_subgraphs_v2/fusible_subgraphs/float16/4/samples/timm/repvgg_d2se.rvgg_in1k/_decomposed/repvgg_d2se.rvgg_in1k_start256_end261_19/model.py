import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
        conv2d = torch.conv2d(in_7, in_5, in_4, (1, 1), (0, 0), (1, 1), 1);  in_7 = in_5 = in_4 = None
        tmp_7 = conv2d.sigmoid();  conv2d = None
        tmp_8 = in_6 * tmp_7;  in_6 = tmp_7 = None
        tmp_9 = torch.nn.functional.relu(tmp_8, inplace = True);  tmp_8 = None
        tmp_10 = torch.nn.functional.batch_norm(tmp_9, in_0, in_1, in_3, in_2, False, 0.1, 1e-05);  in_0 = in_1 = in_3 = in_2 = None
        return (tmp_9, tmp_10)
        