class Program_weight_tensor_meta_query_layer:
	name = "in_0"
	original_name = "query_layer"
	shape = [2, 1, 16384, 64]
	dtype = "torch.float16"
	device = "cuda:0"
	mean = -0.024
	std = 0.338


class Program_weight_tensor_meta_transpose_4:
	name = "in_1"
	original_name = "transpose_4"
	shape = [2, 1, 64, 256]
	dtype = "torch.float16"
	device = "cuda:0"
	mean = 0.078
	std = 0.153


class Program_weight_tensor_meta_value_layer:
	name = "in_2"
	original_name = "value_layer"
	shape = [2, 1, 256, 64]
	dtype = "torch.float16"
	device = "cuda:0"
	mean = 0.001
	std = 0.028
