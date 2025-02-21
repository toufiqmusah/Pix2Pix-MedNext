# models.py
import torch
import torch.nn as nn
from nnunet_mednext import create_mednext_v1
from generative.networks.nets import PatchDiscriminator

class MedNextGenerator3D(nn.Module):
    def __init__(self, input_channels=1, output_channels=1):
        super(MedNextGenerator3D, self).__init__()
        # Initialize the MedNeXt model (make sure the mednext repo is cloned and in your PYTHONPATH)
        self.model = create_mednext_v1(
            num_input_channels=input_channels,
            num_classes=output_channels,
            model_id='M',              
            kernel_size=3,
            deep_supervision=False     
        )
        self.final_activation = nn.Tanh()

    def forward(self, x):
        x = self.model(x)
        return self.final_activation(x)

class Discriminator3D(nn.Module):
    def __init__(self):
        super(Discriminator3D, self).__init__()
        # Using 2 channels (concatenated input and target)
        self.layer1 = self.conv3d_relu(2, 16, kernel_size=5, cnt=1)
        self.layer2 = self.conv3d_relu(16, 32, pool_size=4)
        self.layer3 = self.conv3d_relu(32, 64, pool_size=2)
        self.layer4 = self.conv3d_relu(64, 128, pool_size=2)
        self.layer5 = self.conv3d_relu(128, 256, pool_size=2)
        self.layer6 = nn.Conv3d(256, 1, kernel_size=1)

    def conv3d_relu(self, in_c, out_c, kernel_size=3, pool_size=None, cnt=2):
        layers = []
        for i in range(cnt):
            if i == 0 and pool_size is not None:
                # Downsample width, height and depth
                layers.append(nn.AvgPool3d(pool_size))
            layers.append(nn.Conv3d(
                in_c if i == 0 else out_c,
                out_c,
                kernel_size,
                padding=(kernel_size - 1) // 2
            ))
            layers.append(nn.InstanceNorm3d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        features = []
        x = self.layer1(x)
        features.append(x)
        x = self.layer2(x)
        features.append(x)
        x = self.layer3(x)
        features.append(x)
        x = self.layer4(x)
        features.append(x)
        x = self.layer5(x)
        features.append(x)
        x = self.layer6(x)
        features.append(x)
        return features  

MonaiDiscriminator = PatchDiscriminator(
    spatial_dims = 3,
    num_layers_d=2,
    num_channels=32,
    in_channels=2,
    out_channels=1
)