import torch

from torch import device

from torch import inf

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, w_0 : torch.Tensor, w_1 : torch.Tensor):
        tmp_4 = in_1[0]
        tmp_5 = in_1[1]
        tmp_6 = tmp_4 != tmp_5;  tmp_4 = tmp_5 = None
        tmp_7 = torch.arange(0, 1000, device = device(type='cuda'))
        tmp_8 = tmp_7.view(1, -1);  tmp_7 = None
        tmp_9 = tmp_8.repeat(2, 1);  tmp_8 = None
        tmp_10 = in_1[(slice(None, None, None), tmp_6)];  in_1 = tmp_6 = None
        sym_size_int = torch.ops.aten.sym_size.int(tmp_10, 1)
        _check_is_size_2 = torch._check_is_size(sym_size_int);  _check_is_size_2 = None
        ge_1 = sym_size_int >= 0
        _assert_scalar_default_2 = torch.ops.aten._assert_scalar.default(ge_1, "Runtime assertion failed for expression u0 >= 0 on node 'ge_1'");  ge_1 = _assert_scalar_default_2 = None
        le_1 = sym_size_int <= 100
        _assert_scalar_default_3 = torch.ops.aten._assert_scalar.default(le_1, "Runtime assertion failed for expression u0 <= 100 on node 'le_1'");  le_1 = _assert_scalar_default_3 = None
        _check_is_size = torch._check_is_size(sym_size_int);  _check_is_size = None
        _check_is_size_1 = torch._check_is_size(sym_size_int);  _check_is_size_1 = None
        tmp_18 = torch.cat([tmp_10, tmp_9], dim = 1);  tmp_10 = tmp_9 = None
        sym_sum = torch.sym_sum([1000, sym_size_int]);  sym_size_int = None
        tmp_20 = torch.ones((sym_sum,), dtype = torch.float32, device = device(type='cuda'));  sym_sum = None
        tmp_21 = tmp_18[0]
        tmp_22 = tmp_18[1]
        tmp_23 = tmp_22.view((-1,))
        tmp_24 = tmp_23.expand_as(tmp_20);  tmp_23 = None
        tmp_25 = tmp_20.new_zeros((1000,))
        tmp_26 = tmp_25.scatter_add_(0, tmp_24, tmp_20);  tmp_25 = tmp_24 = None
        tmp_27 = tmp_26.pow_(-0.5);  tmp_26 = None
        tmp_28 = tmp_27.__eq__(inf)
        tmp_29 = tmp_27.masked_fill_(tmp_28, 0);  tmp_28 = tmp_29 = None
        tmp_30 = tmp_27[tmp_21];  tmp_21 = None
        tmp_31 = tmp_30 * tmp_20;  tmp_30 = tmp_20 = None
        tmp_32 = tmp_27[tmp_22];  tmp_27 = tmp_22 = None
        tmp_33 = tmp_31 * tmp_32;  tmp_31 = tmp_32 = None
        linear = torch.nn.functional.linear(in_0, w_0, None);  in_0 = w_0 = None
        tmp_35 = tmp_18[1]
        tmp_36 = tmp_18[0];  tmp_18 = None
        tmp_37 = linear.index_select(-2, tmp_36);  linear = tmp_36 = None
        tmp_38 = tmp_33.view(-1, 1);  tmp_33 = None
        tmp_39 = tmp_38 * tmp_37;  tmp_38 = tmp_37 = None
        tmp_40 = tmp_35.view((-1, 1));  tmp_35 = None
        tmp_41 = tmp_40.expand_as(tmp_39);  tmp_40 = None
        tmp_42 = tmp_39.new_zeros((1000, 16))
        tmp_43 = tmp_42.scatter_add_(0, tmp_41, tmp_39);  tmp_42 = tmp_41 = tmp_39 = None
        tmp_44 = tmp_43 + w_1;  tmp_43 = w_1 = None
        return (tmp_44,)
        