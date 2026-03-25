import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0
        tmp_1 = tmp_0[slice(None, None, None), in_2]
        tmp_0 = None
        tmp_2 = torch.ops.aten.sym_size.int(tmp_1, 1)
        tmp_3 = torch._check_is_size(tmp_2)
        tmp_3 = None
        tmp_4 = tmp_2 >= 0
        tmp_5 = torch.ops.aten._assert_scalar.default(tmp_4, "Runtime assertion failed for expression u0 >= 0 on node 'ge_1'")
        tmp_4 = tmp_5 = None
        tmp_6 = tmp_2 <= 128
        tmp_7 = torch.ops.aten._assert_scalar.default(tmp_6, "Runtime assertion failed for expression u0 <= 128 on node 'le_1'")
        tmp_6 = tmp_7 = None
        tmp_8 = torch._check_is_size(tmp_2)
        tmp_8 = None
        tmp_9 = torch.cat([tmp_1, in_1], dim=1)
        tmp_1 = None
        tmp_10 = torch.sym_sum([128, tmp_2])
        tmp_2 = None
        tmp_11 = torch.ones((tmp_10,), dtype=torch.float32, device=device(type='cuda'))
        tmp_10 = None
        return (tmp_9, tmp_11)