import torch
import torch.nn as nn
import torch.optim as optim
from typing import Sequence


class Skeleton(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_channels: int,
        bottleneck_channels: int,
        stride: int = 1,
        downsample: nn.Module = None,
    ) -> None:
        super().__init__()
        out_channels = bottleneck_channels * self.expansion

        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)

        self.conv2 = nn.Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)

        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class Mohanet(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 3,
        time_feature_dim: int = 4, # new
        layers: Sequence[int] = (3, 4, 6, 3),
        base_width: int = 64,
        stage_strides: Sequence[int] = (1, 2, 2, 2),
        stem_kernel_size: int = 7,
        stem_stride: int = 2,
        use_stem_pool: bool = True,
    ) -> None:
        super().__init__()
        if len(layers) != 4:
            raise ValueError("layers must contain 4 integers for ResNet stages.")
        if len(stage_strides) != 4:
            raise ValueError("stage_strides must contain 4 integers for ResNet stages.")

        self.in_channels = base_width

        stem_padding = stem_kernel_size // 2
        self.conv1 = nn.Conv2d(
            in_channels,
            base_width,
            kernel_size=stem_kernel_size,
            stride=stem_stride,
            padding=stem_padding,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(base_width)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) if use_stem_pool else nn.Identity()

        widths = [base_width, base_width * 2, base_width * 4, base_width * 8]
        self.layer1 = self._make_layer(widths[0], blocks=layers[0], stride=stage_strides[0])
        self.layer2 = self._make_layer(widths[1], blocks=layers[1], stride=stage_strides[1])
        self.layer3 = self._make_layer(widths[2], blocks=layers[2], stride=stage_strides[2])
        self.layer4 = self._make_layer(widths[3], blocks=layers[3], stride=stage_strides[3])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # new
        img_feature_dim = widths[3] * Skeleton.expansion
        combined_dim = img_feature_dim + time_feature_dim
        
        self.fusion_classifier = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )
        self.num_classes = num_classes

        self._init_weights()

    def _make_layer(self, bottleneck_channels: int, blocks: int, stride: int) -> nn.Sequential:
        out_channels = bottleneck_channels * Skeleton.expansion
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = [Skeleton(self.in_channels, bottleneck_channels, stride=stride, downsample=downsample)]
        self.in_channels = out_channels

        for _ in range(1, blocks):
            layers.append(Skeleton(self.in_channels, bottleneck_channels))

        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0.0, 0.01)
                nn.init.constant_(module.bias, 0.0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    # new
    def forward_logits(self, x: torch.Tensor, time_x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        combined_features = torch.cat((features, time_x), dim=1)
        return self.fusion_classifier(combined_features)

    def forward_proba(self, x: torch.Tensor, time_x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward_logits(x, time_x), dim=1)

    def forward(self, x: torch.Tensor, time_x: torch.Tensor) -> torch.Tensor:
        logits = self.forward_logits(x, time_x)
        return torch.argmax(logits, dim=1) + 1


def build_mohanet(
    num_classes: int = 10,
    in_channels: int = 3,
    time_feature_dim: int = 4, # new
    layers: Sequence[int] = (3, 4, 6, 3),
    base_width: int = 64,
    stage_strides: Sequence[int] = (1, 2, 2, 2),
    stem_kernel_size: int = 7,
    stem_stride: int = 2,
    use_stem_pool: bool = True,
) -> Mohanet:
    return Mohanet(
        num_classes=num_classes,
        in_channels=in_channels,
        time_feature_dim=time_feature_dim, # new
        layers=layers,
        base_width=base_width,
        stage_strides=stage_strides,
        stem_kernel_size=stem_kernel_size,
        stem_stride=stem_stride,
        use_stem_pool=use_stem_pool,
    )


def build_training_components(
    model: nn.Module,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    lr_factor: float = 0.1,
    lr_patience: int = 3,
):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=lr_factor,
        patience=lr_patience,
    )
    return optimizer, scheduler

BATCH_SIZE = 256