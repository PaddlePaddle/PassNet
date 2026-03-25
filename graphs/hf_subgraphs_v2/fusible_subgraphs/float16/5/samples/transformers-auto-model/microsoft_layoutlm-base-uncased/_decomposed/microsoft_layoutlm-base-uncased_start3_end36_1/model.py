import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor, in_9 : torch.Tensor, in_10 : torch.Tensor, in_11 : torch.Tensor, in_12 : torch.Tensor, in_13 : torch.Tensor):
        tmp_12 = in_12.to(dtype = torch.float32);  in_12 = None
        tmp_13 = 1.0 - tmp_12;  tmp_12 = None
        tmp_14 = tmp_13 * -3.4028234663852886e+38;  tmp_13 = None
        tmp_15 = in_2[(slice(None, None, None), slice(None, 512, None))];  in_2 = None
        tmp_16 = torch.nn.functional.embedding(in_0, in_9, 0, None, 2.0, False, False);  in_0 = in_9 = None
        tmp_17 = torch.nn.functional.embedding(tmp_15, in_6, None, None, 2.0, False, False);  tmp_15 = in_6 = None
        tmp_18 = in_13[(slice(None, None, None), slice(None, None, None), 0)]
        tmp_19 = torch.nn.functional.embedding(tmp_18, in_10, None, None, 2.0, False, False);  tmp_18 = None
        tmp_20 = in_13[(slice(None, None, None), slice(None, None, None), 1)]
        tmp_21 = torch.nn.functional.embedding(tmp_20, in_11, None, None, 2.0, False, False);  tmp_20 = None
        tmp_22 = in_13[(slice(None, None, None), slice(None, None, None), 2)]
        tmp_23 = torch.nn.functional.embedding(tmp_22, in_10, None, None, 2.0, False, False);  tmp_22 = in_10 = None
        tmp_24 = in_13[(slice(None, None, None), slice(None, None, None), 3)]
        tmp_25 = torch.nn.functional.embedding(tmp_24, in_11, None, None, 2.0, False, False);  tmp_24 = in_11 = None
        tmp_26 = in_13[(slice(None, None, None), slice(None, None, None), 3)]
        tmp_27 = in_13[(slice(None, None, None), slice(None, None, None), 1)]
        tmp_28 = tmp_26 - tmp_27;  tmp_26 = tmp_27 = None
        tmp_29 = torch.nn.functional.embedding(tmp_28, in_5, None, None, 2.0, False, False);  tmp_28 = in_5 = None
        tmp_30 = in_13[(slice(None, None, None), slice(None, None, None), 2)]
        tmp_31 = in_13[(slice(None, None, None), slice(None, None, None), 0)];  in_13 = None
        tmp_32 = tmp_30 - tmp_31;  tmp_30 = tmp_31 = None
        tmp_33 = torch.nn.functional.embedding(tmp_32, in_8, None, None, 2.0, False, False);  tmp_32 = in_8 = None
        tmp_34 = torch.nn.functional.embedding(in_1, in_7, None, None, 2.0, False, False);  in_1 = in_7 = None
        tmp_35 = tmp_16 + tmp_17;  tmp_16 = tmp_17 = None
        tmp_36 = tmp_35 + tmp_19;  tmp_35 = tmp_19 = None
        tmp_37 = tmp_36 + tmp_21;  tmp_36 = tmp_21 = None
        tmp_38 = tmp_37 + tmp_23;  tmp_37 = tmp_23 = None
        tmp_39 = tmp_38 + tmp_25;  tmp_38 = tmp_25 = None
        tmp_40 = tmp_39 + tmp_29;  tmp_39 = tmp_29 = None
        tmp_41 = tmp_40 + tmp_33;  tmp_40 = tmp_33 = None
        tmp_42 = tmp_41 + tmp_34;  tmp_41 = tmp_34 = None
        tmp_43 = torch.nn.functional.layer_norm(tmp_42, (768,), in_4, in_3, 1e-12);  tmp_42 = in_4 = in_3 = None
        tmp_44 = torch.nn.functional.dropout(tmp_43, 0.1, False, False);  tmp_43 = None
        return (tmp_44, tmp_14)
        