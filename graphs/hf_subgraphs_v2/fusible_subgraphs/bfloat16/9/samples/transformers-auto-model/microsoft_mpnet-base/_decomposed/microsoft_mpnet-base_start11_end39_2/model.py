import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor):
        tmp_5 = torch.nn.functional.embedding(in_0, in_4, 1, None, 2.0, False, False);  in_0 = in_4 = None
        tmp_6 = torch.nn.functional.embedding(in_5, in_3, 1, None, 2.0, False, False);  in_5 = in_3 = None
        tmp_7 = tmp_5 + tmp_6;  tmp_5 = tmp_6 = None
        tmp_8 = torch.nn.functional.layer_norm(tmp_7, (768,), in_2, in_1, 1e-05);  tmp_7 = in_2 = in_1 = None
        tmp_9 = torch.nn.functional.dropout(tmp_8, 0.1, False, False);  tmp_8 = None
        tmp_10 = torch.arange(11, dtype = torch.int64)
        tmp_11 = tmp_10[(slice(None, None, None), None)];  tmp_10 = None
        tmp_12 = torch.arange(11, dtype = torch.int64)
        tmp_13 = tmp_12[(None, slice(None, None, None))];  tmp_12 = None
        tmp_14 = tmp_13 - tmp_11;  tmp_13 = tmp_11 = None
        tmp_15 = -tmp_14;  tmp_14 = None
        tmp_16 = tmp_15 < 0
        tmp_17 = tmp_16.to(torch.int64);  tmp_16 = None
        tmp_18 = tmp_17 * 16;  tmp_17 = None
        tmp_19 = 0 + tmp_18;  tmp_18 = None
        tmp_20 = torch.abs(tmp_15);  tmp_15 = None
        tmp_21 = tmp_20 < 8
        tmp_22 = tmp_20.float()
        tmp_23 = tmp_22 / 8;  tmp_22 = None
        tmp_24 = torch.log(tmp_23);  tmp_23 = None
        tmp_25 = tmp_24 / 2.772588722239781;  tmp_24 = None
        tmp_26 = tmp_25 * 8;  tmp_25 = None
        tmp_27 = tmp_26.to(torch.int64);  tmp_26 = None
        tmp_28 = 8 + tmp_27;  tmp_27 = None
        tmp_29 = torch.full_like(tmp_28, 15)
        tmp_30 = torch.min(tmp_28, tmp_29);  tmp_28 = tmp_29 = None
        tmp_31 = torch.where(tmp_21, tmp_20, tmp_30);  tmp_21 = tmp_20 = tmp_30 = None
        tmp_19 += tmp_31;  tmp_32 = tmp_19;  tmp_19 = tmp_31 = None
        return (tmp_9, tmp_32)
        