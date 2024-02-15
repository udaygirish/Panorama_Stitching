# Spatial Transformer from scratch - Adapted from 2 sources in Github
import torch
import numpy as np


def spatial_transformer(U, theta, out_size):
    """Spatial Transformer Layer
    Parameters
    U: float (The output of a conv net
    [batch, height, width, channel])
    theta: float
    The output of the localization network [num_batch, 6]
    out_size : tuple of two ints
    The size of the output of the network (height, width)
    Here in our case - (128,128)

    References:
    Spatial Transformer Networks
    Max Jaderberg, Karen Simonyan , Andrew Zisserma et.al

    https://github.com/skaae/transformer_network/blob/master/transformerlayer.py
    https://github.com/breadcake/unsupervisedDeepHomography-pytorch
    """

    def _repeat(x, n_repeats):
        rep = torch.ones(
            [
                n_repeats,
            ]
        ).unsqueeze(0)
        rep = rep.int()
        x = x.int()

        x = torch.matmul(x.reshape([-1, 1]), rep)
        return x.reshape([-1])

    def _interpolate(im, x, y, out_size, scale_h):
        num_batch, channels, height, width = im.size()

        height_f = float(height)
        width_f = float(width)
        out_height = out_size[0]
        out_width = out_size[1]

        zero = 0
        max_y = height - 1
        max_x = width - 1

        if scale_h:
            x = (x + 1.0) * (width_f) / 2.0
            y = (y + 1.0) * (height_f) / 2.0

        # Sampling
        x0 = torch.floor(x).int()
        x1 = x0 + 1
        y0 = torch.floor(y).int()
        y1 = y0 + 1

        x0 = torch.clamp(x0, zero, max_x)
        x1 = torch.clamp(x1, zero, max_x)
        y0 = torch.clamp(y0, zero, max_y)
        y1 = torch.clamp(y1, zero, max_y)
        dim2 = torch.from_numpy(np.array(width))
        dim1 = torch.from_numpy(np.array(width * height))

        base = _repeat(torch.arange(0, num_batch) * dim1, out_height * out_width)

        if torch.cuda.is_available():
            base = base.cuda()
            dim2 = dim2.cuda()
            dim1 = dim1.cuda()
            x0 = x0.cuda()
            x1 = x1.cuda()
            y0 = y0.cuda()
            y1 = y1.cuda()

        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2

        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # Channels for the interpolation
        im = im.permute(0, 2, 3, 1)
        im_flat = im.reshape([-1, channels])

        idx_a = idx_a.unsqueeze(-1).long()
        idx_b = idx_b.unsqueeze(-1).long()
        idx_c = idx_c.unsqueeze(-1).long()
        idx_d = idx_d.unsqueeze(-1).long()

        # Expand dim
        idx_a = idx_a.expand(height * width * num_batch, channels)
        idx_b = idx_b.expand(height * width * num_batch, channels)
        idx_c = idx_c.expand(height * width * num_batch, channels)
        idx_d = idx_d.expand(height * width * num_batch, channels)

        Ia = torch.gather(im_flat, 0, idx_a)
        Ib = torch.gather(im_flat, 0, idx_b)
        Ic = torch.gather(im_flat, 0, idx_c)
        Id = torch.gather(im_flat, 0, idx_d)

        x0_f = x0.float()
        x1_f = x1.float()
        y0_f = y0.float()
        y1_f = y1.float()

        wa = ((x1_f - x) * (y1_f - y)).unsqueeze(-1)
        wb = ((x1_f - x) * (y - y0_f)).unsqueeze(-1)
        wc = ((x - x0_f) * (y1_f - y)).unsqueeze(-1)
        wd = ((x - x0_f) * (y - y0_f)).unsqueeze(-1)

        output = wa * Ia + wb * Ib + wc * Ic + wd * Id
        return output

    # Construct a meshgrid
    def _meshgrid(height, width, scale_h):
        if scale_h:
            x_t = torch.matmul(
                torch.ones([height, 1]),
                torch.transpose(
                    torch.unsqueeze(torch.linspace(-1.0, 1.0, width), 1), 1, 0
                ),
            )
            y_t = torch.matmul(
                torch.unsqueeze(torch.linspace(-1.0, 1.0, height), 1),
                torch.ones([1, width]),
            )
        else:
            x_t = torch.matmul(
                torch.ones([height, 1]),
                torch.transpose(
                    torch.unsqueeze(torch.linspace(0.0, width.float(), width), 1), 1, 0
                ),
            )
            y_t = torch.matmul(
                torch.unsqueeze(torch.linspace(0.0, height.float(), height), 1),
                torch.ones([1, width]),
            )

        x_t_flat = x_t.reshape([1, -1]).float()
        y_t_flat = y_t.reshape([1, -1]).float()

        ones = torch.ones_like(x_t_flat)
        grid = torch.cat([x_t_flat, y_t_flat, ones], 0)
        if torch.cuda.is_available():
            grid = grid.cuda()
        return grid

    def _transform(theta, input_dim, out_size, scale_h):
        num_batch, num_channels, height, width = input_dim.size()
        theta = theta.reshape([-1, 3, 3]).float()

        out_height = out_size[0]
        out_width = out_size[1]

        grid = _meshgrid(out_height, out_width, scale_h)
        grid = grid.unsqueeze(0).reshape([1, -1])
        shape = grid.size()

        grid = grid.expand(num_batch, shape[1])
        grid = grid.reshape([num_batch, 3, -1])

        T_g = torch.matmul(theta, grid)
        x_s = T_g[:, 0, :]
        y_s = T_g[:, 1, :]
        t_s = T_g[:, 2, :]

        t_s_flat = t_s.reshape([-1])

        # Understand this smaller and smallers
        small = 1e-7
        smallers = 1e-6 * (1 - torch.ge(torch.abs(t_s_flat), small).float())

        t_s_flat = t_s_flat + smallers
        condition = torch.sum(torch.gt(torch.abs(t_s_flat), small)).float()

        x_s_flat = x_s.reshape([-1]) / t_s_flat
        y_s_flat = y_s.reshape([-1]) / t_s_flat

        input_transformed = _interpolate(
            input_dim, x_s_flat, y_s_flat, out_size, scale_h
        )
        output = input_transformed.reshape(
            [num_batch, out_height, out_width, num_channels]
        )
        output = output.permute(0, 3, 1, 2)
        return output, condition

    img_w = U.size()[2]
    img_h = U.size()[3]

    scale_h = True
    output, condition = _transform(theta, U, out_size, scale_h)
    return output, condition
