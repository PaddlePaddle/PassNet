import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5):
        conv1d = torch.conv1d(in_4, in_5, in_2, (1,), (64,), (1,), 16);  in_4 = in_5 = in_2 = None
        tmp_4 = conv1d[(slice(None, None, None), slice(None, None, None), slice(None, -1, None))];  conv1d = None
        tmp_5 = torch.nn.functional.gelu(tmp_4);  tmp_4 = None
        tmp_6 = tmp_5.transpose(1, 2);  tmp_5 = None
        tmp_7 = in_3 + tmp_6;  in_3 = tmp_6 = None
        tmp_8 = torch.nn.functional.dropout(tmp_7, 0.05, False, False);  tmp_7 = None
        tmp_9 = torch.rand([]);  tmp_9 = None
        tmp_10 = torch.nn.functional.layer_norm(tmp_8, (1024,), in_1, in_0, 1e-05);  in_1 = in_0 = None
        return (tmp_8, tmp_10)
        