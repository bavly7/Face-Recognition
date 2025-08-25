import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):

	def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
		super().__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
		self.bn = nn.BatchNorm2d(out_channels)
		self.relu = nn.ReLU(inplace=True)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.conv(x)
		x = self.bn(x)
		x = self.relu(x)
		return x


class EmbeddingNet(nn.Module):

	def __init__(self, embedding_dim: int = 128):
		super().__init__()
		# Input: 3x160x160
		self.stem = nn.Sequential(
			ConvBlock(3, 32, 3, 2, 1),   # 32x80x80
			ConvBlock(32, 64, 3, 1, 1),  # 64x80x80
			nn.MaxPool2d(2),              # 64x40x40
			ConvBlock(64, 128, 3, 1, 1), # 128x40x40
			nn.MaxPool2d(2),              # 128x20x20
			ConvBlock(128, 256, 3, 1, 1),# 256x20x20
			nn.MaxPool2d(2),              # 256x10x10
			ConvBlock(256, 256, 3, 1, 1),# 256x10x10
		)

		self.head = nn.Sequential(
			nn.AdaptiveAvgPool2d(1),
			nn.Flatten(),
			nn.Linear(256, embedding_dim),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.stem(x)
		x = self.head(x)
		x = F.normalize(x, p=2, dim=1)
		return x


def build_model(embedding_dim: int = 128) -> nn.Module:
	return EmbeddingNet(embedding_dim=embedding_dim)

