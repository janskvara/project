import torch
import torch.nn as nn

class CNNetwork(nn.Module):
    def __init__(self, out_dim: int, pool=False):
        """Initialization."""
        super(CNNetwork, self).__init__()

        nf = 32
        s = 1 if pool else 2
        self.layers =  nn.Sequential(
            # (B, 3, 32, 32) images
            self.build_cnn_layer(3, nf),  # (B, nf, 32, 32)
            self.build_cnn_layer(nf, 2 * nf, stride=s, pool=pool),  # (B, 2 * nf, 16, 16)
            self.build_cnn_layer(2 * nf, 4 * nf, stride=s, pool=pool),  # (B, 4 * nf, 8, 8)
            self.build_cnn_layer(4 * nf, 8 * nf, stride=s, pool=pool),  # (B, 8 * nf, 4, 4)
            nn.Flatten(), # (B, 8 * nf * 4 * 4)
            nn.Linear(8 * nf * 4 * 4, 512), nn.ReLU(True), nn.BatchNorm1d(512), # (B, 512)
            nn.Linear(512, out_dim), # (B, 10) logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        return self.layers(x)
    
    def build_cnn_layer(self, in_channels, out_channels, kernal_size=3, stride=1, padding=1, 
                    use_batchnorm=True, pool=False):

        '''
        in_channels: input channels to 2d convolution layer
        out_channels: ouput channels to 2d convolution layer
        kernal_size: kernal size (refer torch conv2d)
        stride: stride (refer torch conv2d)
        padding: padding (refer torch conv2d)  
        use_batchnorm: enable/disable batchnorm (refer torch BatchNorm2d)
        pool: enable/disable pooling (refer torch MaxPool2D)

        should return the built CNN layer
        '''
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernal_size, stride=stride, padding=padding, ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity(),
            nn.MaxPool2d(2, 2) if pool else nn.Identity(),
        )