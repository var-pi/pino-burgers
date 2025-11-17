import torch

def burgers_res(y, nu=0.1):
	_, _, n_time, n_space = y.shape
	dt = 1.0 / (n_time - 1)
	dx = 1.0 / n_space

	ut = torch.gradient(y, dim=2, spacing=dt, edge_order=2)[0]

	u_right = torch.roll(y, -1, dims=3)
	u_left = torch.roll(y, 1, dims=3)
	ux = (u_right - u_left) / (2 * dx)
	uxx = (u_right - 2 * y + u_left) / (dx ** 2)

	return ut + y * ux - nu * uxx  # (17, 16)
