import torch
from einops import rearrange

# Tạo 2 tensor độc lập kích thước [3, 2]
x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
y = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
# x_tmp = rearrange(x, "... d -> ... d 1")
# y_tmp = rearrange(y, "... (d two) -> ... d two", two = 2)
x_pair = rearrange(x, "... (d two) -> ... d two", two = 2)
x_1 = x_pair[..., 0]
x_2 = x_pair[..., 1]
x_rotate = rearrange([x_1, -x_2], "two ... d -> ... d two")
mask_tensor = torch.tril(torch.ones((5, 5), dtype=torch.bool))
# print(x_1)
# print(x_2)
print(mask_tensor)
# print(mask_tensor.shape[-2])
# print(mask_tensor.shape[-1])   

# print(torch.tensor([1, 2])*x)
# res = y_tmp * x_tmp
# print(res)

