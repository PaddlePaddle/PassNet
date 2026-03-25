import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, in_0, in_1):
        conv2d = torch.conv2d(in_1, w_2, w_1, (1, 1), (0, 0), (1, 1), 1);  in_1 = w_2 = w_1 = None
        tmp_10 = torch.nn.functional.gelu(conv2d, approximate = 'none');  conv2d = None
        tmp_11 = torch.nn.functional.dropout(tmp_10, 0.0, False, False);  tmp_10 = None
        conv2d_1 = torch.conv2d(tmp_11, w_4, w_3, (1, 1), (0, 0), (1, 1), 1);  tmp_11 = w_4 = w_3 = None
        tmp_13 = torch.nn.functional.dropout(conv2d_1, 0.0, False, False);  conv2d_1 = None
        tmp_14 = tmp_13 * w_0;  tmp_13 = w_0 = None
        tmp_15 = in_0 + tmp_14;  in_0 = tmp_14 = None
        tmp_16 = torch.nn.functional.batch_norm(tmp_15, w_5, w_6, w_8, w_7, False, 0.1, 1e-05);  w_5 = w_6 = w_8 = w_7 = None
        return (tmp_16, tmp_15)
        