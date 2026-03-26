import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4):
        conv1d = torch.conv1d(in_3, in_4, in_2, (2,), (15,), (1,), 16);  in_4 = in_2 = None
        tmp_4 = torch.nn.functional.gelu(conv1d);  conv1d = None
        tmp_5 = torch.avg_pool1d(in_3, (2,), (2,), (0,), False, True);  in_3 = None
        tmp_6 = tmp_5[(Ellipsis, slice(None, 124, None))];  tmp_5 = None
        tmp_7 = tmp_4[(Ellipsis, slice(None, 124, None))];  tmp_4 = None
        tmp_8 = tmp_6 + tmp_7;  tmp_6 = tmp_7 = None
        tmp_9 = tmp_8.transpose(1, 2);  tmp_8 = None
        tmp_10 = torch.nn.functional.layer_norm(tmp_9, (768,), in_1, in_0, 1e-05);  tmp_9 = in_1 = in_0 = None
        tmp_11 = torch.nn.functional.dropout(tmp_10, 0.1, False, False);  tmp_10 = None
        tmp_12 = torch.rand([]);  tmp_12 = None
        return (tmp_11,)
        