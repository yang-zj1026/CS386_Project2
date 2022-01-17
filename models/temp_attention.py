import torch.nn as nn
from models.dsam_layers import center_crop


class dsam_score_dsn(nn.Module):
    def __init__(self, prev_layer, prev_nfilters, prev_nsamples):
        super(dsam_score_dsn, self).__init__()
        i = prev_layer
        self.avgpool = nn.AvgPool3d((prev_nsamples, 1, 1), stride=1)
        # Make the layers of the preparation step
        self.side_prep = nn.Conv2d(prev_nfilters, 16, kernel_size=3, padding=1)
        # Make the layers of the score_dsn step
        self.score_dsn = nn.Conv2d(16, 1, kernel_size=1, padding=0)
        self.upscale_ = nn.ConvTranspose2d(1, 1, kernel_size=2 ** (1 + i), stride=2 ** i, bias=False)
        self.upscale = nn.ConvTranspose2d(16, 16, kernel_size=2 ** (1 + i), stride=2 ** i, bias=False)

    def forward(self, x, crop_h, crop_w):
        self.crop_h = crop_h
        self.crop_w = crop_w
        x = self.avgpool(x).squeeze(2)
        side_temp = self.side_prep(x)
        side = center_crop(self.upscale(side_temp), self.crop_h, self.crop_w)
        side_out_tmp = self.score_dsn(side_temp)
        side_out = center_crop(self.upscale_(side_out_tmp), self.crop_h, self.crop_w)
        return side, side_out, side_out_tmp
