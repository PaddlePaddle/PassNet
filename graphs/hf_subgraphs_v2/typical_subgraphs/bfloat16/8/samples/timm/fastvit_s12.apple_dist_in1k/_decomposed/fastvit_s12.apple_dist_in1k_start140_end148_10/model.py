import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10):
        conv2d = torch.conv2d(in_10, in_2, in_1, (1, 1), (0, 0), (1, 1), 1);  in_10 = in_2 = in_1 = None
        tmp_10 = torch.nn.functional.gelu(conv2d, approximate = 'none');  conv2d = None
        tmp_11 = torch.nn.functional.dropout(tmp_10, 0.0, False, False);  tmp_10 = None
        conv2d_1 = torch.conv2d(tmp_11, in_4, in_3, (1, 1), (0, 0), (1, 1), 1);  tmp_11 = in_4 = in_3 = None
        tmp_13 = torch.nn.functional.dropout(conv2d_1, 0.0, False, False);  conv2d_1 = None
        tmp_14 = tmp_13 * in_0;  tmp_13 = in_0 = None
        tmp_15 = in_9 + tmp_14;  in_9 = tmp_14 = None
        tmp_16 = torch.nn.functional.batch_norm(tmp_15, in_5, in_6, in_8, in_7, False, 0.1, 1e-05);  in_5 = in_6 = in_8 = in_7 = None
        return (tmp_16, tmp_15)
        