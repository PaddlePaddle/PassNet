import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, in_0, in_1):
        conv2d = torch.conv2d(in_1, w_2, w_1, (1, 1), (0, 0), (1, 1), 1);  in_1 = w_2 = w_1 = None
        tmp_6 = torch.nn.functional.gelu(conv2d, approximate = 'none');  conv2d = None
        tmp_7 = torch.nn.functional.dropout(tmp_6, 0.0, False, False);  tmp_6 = None
        conv2d_1 = torch.conv2d(tmp_7, w_4, w_3, (1, 1), (0, 0), (1, 1), 1);  tmp_7 = w_4 = w_3 = None
        tmp_9 = torch.nn.functional.dropout(conv2d_1, 0.0, False, False);  conv2d_1 = None
        tmp_10 = tmp_9 * w_0;  tmp_9 = w_0 = None
        tmp_11 = in_0 + tmp_10;  in_0 = tmp_10 = None
        return (tmp_11,)
        