import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8):
        conv2d = torch.conv2d(in_8, in_2, in_1, (1, 1), (0, 0), (1, 1), 1);  in_8 = in_2 = in_1 = None
        tmp_8 = torch.nn.functional.dropout(conv2d, 0.0, False, False);  conv2d = None
        tmp_9 = tmp_8 * in_0;  tmp_8 = in_0 = None
        tmp_10 = in_7 + tmp_9;  in_7 = tmp_9 = None
        tmp_11 = torch.nn.functional.batch_norm(tmp_10, in_3, in_4, in_6, in_5, False, 0.1, 1e-05);  in_3 = in_4 = in_6 = in_5 = None
        return (tmp_11, tmp_10)
        