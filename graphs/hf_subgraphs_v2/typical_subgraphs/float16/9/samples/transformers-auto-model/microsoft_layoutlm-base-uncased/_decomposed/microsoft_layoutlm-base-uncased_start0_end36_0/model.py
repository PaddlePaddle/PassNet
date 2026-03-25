import torch

from torch import device

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, w_7 : torch.Tensor, w_8 : torch.Tensor, w_9 : torch.Tensor):
        tmp_13 = torch.zeros((1, 11, 4), dtype = torch.int64, device = device(type='cuda', index=0))
        tmp_14 = in_0.unsqueeze(1);  in_0 = None
        tmp_15 = tmp_14.unsqueeze(2);  tmp_14 = None
        tmp_16 = tmp_15.to(dtype = torch.float32);  tmp_15 = None
        tmp_17 = 1.0 - tmp_16;  tmp_16 = None
        tmp_18 = tmp_17 * -3.4028234663852886e+38;  tmp_17 = None
        tmp_19 = w_0[(slice(None, None, None), slice(None, 11, None))];  w_0 = None
        tmp_20 = torch.nn.functional.embedding(in_1, w_7, 0, None, 2.0, False, False);  in_1 = w_7 = None
        tmp_21 = torch.nn.functional.embedding(tmp_19, w_4, None, None, 2.0, False, False);  tmp_19 = w_4 = None
        tmp_22 = tmp_13[(slice(None, None, None), slice(None, None, None), 0)]
        tmp_23 = torch.nn.functional.embedding(tmp_22, w_8, None, None, 2.0, False, False);  tmp_22 = None
        tmp_24 = tmp_13[(slice(None, None, None), slice(None, None, None), 1)]
        tmp_25 = torch.nn.functional.embedding(tmp_24, w_9, None, None, 2.0, False, False);  tmp_24 = None
        tmp_26 = tmp_13[(slice(None, None, None), slice(None, None, None), 2)]
        tmp_27 = torch.nn.functional.embedding(tmp_26, w_8, None, None, 2.0, False, False);  tmp_26 = w_8 = None
        tmp_28 = tmp_13[(slice(None, None, None), slice(None, None, None), 3)]
        tmp_29 = torch.nn.functional.embedding(tmp_28, w_9, None, None, 2.0, False, False);  tmp_28 = w_9 = None
        tmp_30 = tmp_13[(slice(None, None, None), slice(None, None, None), 3)]
        tmp_31 = tmp_13[(slice(None, None, None), slice(None, None, None), 1)]
        tmp_32 = tmp_30 - tmp_31;  tmp_30 = tmp_31 = None
        tmp_33 = torch.nn.functional.embedding(tmp_32, w_3, None, None, 2.0, False, False);  tmp_32 = w_3 = None
        tmp_34 = tmp_13[(slice(None, None, None), slice(None, None, None), 2)]
        tmp_35 = tmp_13[(slice(None, None, None), slice(None, None, None), 0)];  tmp_13 = None
        tmp_36 = tmp_34 - tmp_35;  tmp_34 = tmp_35 = None
        tmp_37 = torch.nn.functional.embedding(tmp_36, w_6, None, None, 2.0, False, False);  tmp_36 = w_6 = None
        tmp_38 = torch.nn.functional.embedding(in_2, w_5, None, None, 2.0, False, False);  in_2 = w_5 = None
        tmp_39 = tmp_20 + tmp_21;  tmp_20 = tmp_21 = None
        tmp_40 = tmp_39 + tmp_23;  tmp_39 = tmp_23 = None
        tmp_41 = tmp_40 + tmp_25;  tmp_40 = tmp_25 = None
        tmp_42 = tmp_41 + tmp_27;  tmp_41 = tmp_27 = None
        tmp_43 = tmp_42 + tmp_29;  tmp_42 = tmp_29 = None
        tmp_44 = tmp_43 + tmp_33;  tmp_43 = tmp_33 = None
        tmp_45 = tmp_44 + tmp_37;  tmp_44 = tmp_37 = None
        tmp_46 = tmp_45 + tmp_38;  tmp_45 = tmp_38 = None
        tmp_47 = torch.nn.functional.layer_norm(tmp_46, (768,), w_2, w_1, 1e-12);  tmp_46 = w_2 = w_1 = None
        tmp_48 = torch.nn.functional.dropout(tmp_47, 0.1, False, False);  tmp_47 = None
        return (tmp_48, tmp_18)
        