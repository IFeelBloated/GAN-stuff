import math
import torch
import torch.nn as nn

def LearningRateEqualizer(LinearLayer):
    def Equalizer(Layer, _):
        Weight = Layer.UnnormalizedWeight
        FanIn = Weight.data.size(1) * Weight.data[0][0].numel()
        Layer.weight = Weight * math.sqrt(2 / FanIn)
        
    LinearLayer.weight.data.normal_()
    LinearLayer.bias.data.zero_()
    
    Weight = LinearLayer.weight
    del LinearLayer._parameters['weight']
    LinearLayer.register_parameter('UnnormalizedWeight', nn.Parameter(Weight.data))
    LinearLayer.register_forward_pre_hook(Equalizer)
    
    return LinearLayer

def SpectralNormalizer(LinearLayer, *args, **kw):
    LinearLayer.weight.data.normal_()
    LinearLayer.bias.data.zero_()
    return nn.utils.spectral_norm(LinearLayer, *args, **kw)

def MinibatchStandardDeviation(x):
    StandardDeviation = torch.sqrt(torch.mean((x - torch.mean(x, dim=0, keepdim=True)) ** 2, dim=0, keepdim=True) + 1e-8)
    return torch.cat([x, torch.mean(StandardDeviation, dim=1, keepdim=True).expand(x.size(0), -1, -1, -1)], 1)

class GeneratorBlock(nn.Module):
      def __init__(self, InputChannels, OutputChannels, ReceptiveField=3):
          super(GeneratorBlock, self).__init__()
          
          CompressedChannels = InputChannels // 4
          self.OutputChannels = OutputChannels
          
          self.LinearLayer1 = LearningRateEqualizer(nn.Conv2d(InputChannels, CompressedChannels * 4, kernel_size=ReceptiveField, stride=1, padding=(ReceptiveField - 1) // 2, bias=True))
          self.LinearLayer2 = LearningRateEqualizer(nn.Conv2d(CompressedChannels, OutputChannels, kernel_size=ReceptiveField, stride=1, padding=(ReceptiveField - 1) // 2, bias=True))
          
      def forward(self, x):
          y = nn.functional.leaky_relu(x, 0.2)
          y = self.LinearLayer1(y)
          
          y = nn.functional.pixel_shuffle(y, 2)
          
          y = nn.functional.leaky_relu(y, 0.2, inplace=True)
          y = self.LinearLayer2(y)
          
          return nn.functional.interpolate(x[:, :self.OutputChannels, :, :], scale_factor=2, mode='bilinear', align_corners=False) + y, y

class DiscriminatorBlock(nn.Module):
      def __init__(self, InputChannels, OutputChannels, ReceptiveField=3):
          super(DiscriminatorBlock, self).__init__()
          
          CompressedChannels = OutputChannels // 4
          
          self.LinearLayer1 = SpectralNormalizer(nn.Conv2d(InputChannels + 1, CompressedChannels, kernel_size=ReceptiveField, stride=1, padding=(ReceptiveField - 1) // 2, bias=True))
          self.LinearLayer2 = SpectralNormalizer(nn.Conv2d(CompressedChannels * 4, OutputChannels, kernel_size=ReceptiveField, stride=1, padding=(ReceptiveField - 1) // 2, bias=True))
          
          if InputChannels != OutputChannels:
              self.ShortcutLayer = SpectralNormalizer(nn.Conv2d(InputChannels, OutputChannels - InputChannels, kernel_size=1, stride=1, padding=0, bias=True))
          
      def forward(self, x):
          y = nn.functional.leaky_relu(x, 0.2, inplace=True)
          y = self.LinearLayer1(MinibatchStandardDeviation(y))

          y = nn.functional.pixel_unshuffle(y, 2)
          
          y = nn.functional.leaky_relu(y, 0.2, inplace=True)
          y = self.LinearLayer2(y)
          
          x = nn.functional.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False, recompute_scale_factor=True)
          if hasattr(self, 'ShortcutLayer'):
              x = torch.cat([x, self.ShortcutLayer(nn.functional.leaky_relu(x, 0.2, inplace=True))], 1)

          return x + y
      
class GeneratorOpeningBlock(nn.Module):
      def __init__(self, LatentDimension, OutputChannels, ReceptiveField=3):
          super(GeneratorOpeningBlock, self).__init__()
          
          self.CompressedChannels = LatentDimension // 4
          
          self.LinearLayer1 = LearningRateEqualizer(nn.Linear(LatentDimension, self.CompressedChannels * 4 * 4))
          self.LinearLayer2 = LearningRateEqualizer(nn.Conv2d(self.CompressedChannels, OutputChannels, kernel_size=ReceptiveField, stride=1, padding=(ReceptiveField - 1) // 2, bias=True))
          
      def forward(self, x):
          x = x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + 1e-8) # spherical latent space
          y = self.LinearLayer1(x).view(-1, self.CompressedChannels * 4 * 4, 1, 1)
          
          y = nn.functional.pixel_shuffle(y, 4)
          
          y = nn.functional.leaky_relu(y, 0.2, inplace=True)
          
          return self.LinearLayer2(y)
      
class DiscriminatorClosingBlock(nn.Module):
      def __init__(self, InputChannels, LatentDimension, ReceptiveField=3):
          super(DiscriminatorClosingBlock, self).__init__()
          
          self.CompressedChannels = LatentDimension // 4
          
          self.LinearLayer1 = SpectralNormalizer(nn.Conv2d(InputChannels + 1, self.CompressedChannels, kernel_size=ReceptiveField, stride=1, padding=(ReceptiveField - 1) // 2, bias=True))
          self.LinearLayer2 = SpectralNormalizer(nn.Linear(self.CompressedChannels * 4 * 4, LatentDimension))
          
      def forward(self, x):
          y = nn.functional.leaky_relu(x, 0.2, inplace=True)
          y = self.LinearLayer1(MinibatchStandardDeviation(y))
          
          y = nn.functional.pixel_unshuffle(y, 4).view(-1, self.CompressedChannels * 4 * 4)
          
          y = nn.functional.leaky_relu(y, 0.2, inplace=True)
          
          return self.LinearLayer2(y)
      
class ToRGB(nn.Module):
      def __init__(self, InputChannels):
          super(ToRGB, self).__init__()
          
          self.LinearLayer = LearningRateEqualizer(nn.Conv2d(InputChannels, 3, kernel_size=1, stride=1, padding=0, bias=True))

      def forward(self, x):
          x = nn.functional.leaky_relu(x, 0.2, inplace=True)
          return self.LinearLayer(x)
      
def FromRGB(OutputChannels):
    return SpectralNormalizer(nn.Conv2d(3, OutputChannels, kernel_size=1, stride=1, padding=0, bias=True))
  
class Generator(nn.Module):
    def __init__(self, LatentDimension):
        super(Generator, self).__init__()
        
        self.Layer4x4 = GeneratorOpeningBlock(LatentDimension, 512 * 2)
        self.ToRGB4x4 = ToRGB(512 * 2)
        
        self.Layer8x8 = GeneratorBlock(512 * 2, 512 * 2)
        self.ToRGB8x8 = ToRGB(512 * 2)
        
        self.Layer16x16 = GeneratorBlock(512 * 2, 512)
        self.ToRGB16x16 = ToRGB(512)
        
        self.Layer32x32 = GeneratorBlock(512, 512)
        self.ToRGB32x32 = ToRGB(512)
        
        self.Layer64x64 = GeneratorBlock(512, 512)
        self.ToRGB64x64 = ToRGB(512)
        
        self.Layer128x128 = GeneratorBlock(512, 256)
        self.ToRGB128x128 = ToRGB(256)
      
    def forward(self, x):
        x = self.Layer4x4(x)
        Output4x4 = self.ToRGB4x4(x)

        x, Residual = self.Layer8x8(x)
        Output8x8 = nn.functional.interpolate(Output4x4, scale_factor=2, mode='bilinear', align_corners=False) + self.ToRGB8x8(Residual)

        x, Residual = self.Layer16x16(x)
        Output16x16 = nn.functional.interpolate(Output8x8, scale_factor=2, mode='bilinear', align_corners=False) + self.ToRGB16x16(Residual)

        x, Residual = self.Layer32x32(x)
        Output32x32 = nn.functional.interpolate(Output16x16, scale_factor=2, mode='bilinear', align_corners=False) + self.ToRGB32x32(Residual)

        x, Residual = self.Layer64x64(x)
        Output64x64 = nn.functional.interpolate(Output32x32, scale_factor=2, mode='bilinear', align_corners=False) + self.ToRGB64x64(Residual)

        x, Residual = self.Layer128x128(x)
        Output128x128 = nn.functional.interpolate(Output64x64, scale_factor=2, mode='bilinear', align_corners=False) + self.ToRGB128x128(Residual)

        return torch.tanh(Output4x4), torch.tanh(Output8x8), torch.tanh(Output16x16), torch.tanh(Output32x32), torch.tanh(Output64x64), torch.tanh(Output128x128)

class Discriminator(nn.Module):
    def __init__(self, LatentDimension):
        super(Discriminator, self).__init__()
        
        self.FromRGB128x128 = FromRGB(256)
        self.Layer128x128 = DiscriminatorBlock(256, 512)
        
        self.FromRGB64x64 = FromRGB(512)
        self.Layer64x64 = DiscriminatorBlock(512, 512)
        
        self.FromRGB32x32 = FromRGB(512)
        self.Layer32x32 = DiscriminatorBlock(512, 512)
        
        self.FromRGB16x16 = FromRGB(512)
        self.Layer16x16 = DiscriminatorBlock(512, 512 * 2)
        
        self.FromRGB8x8 = FromRGB(512 * 2)
        self.Layer8x8 = DiscriminatorBlock(512 * 2, 512 * 2)
        
        self.FromRGB4x4 = FromRGB(512 * 2)
        self.Layer4x4 = DiscriminatorClosingBlock(512 * 2, LatentDimension)
        
        self.CriticLayer = SpectralNormalizer(nn.Linear(LatentDimension, 1))
        
    def forward(self, Input4x4, Input8x8, Input16x16, Input32x32, Input64x64, Input128x128):
        x128 = self.FromRGB128x128(Input128x128)
        x128 = self.Layer128x128(x128)
        
        x64 = self.FromRGB64x64(Input64x64)
        x64 = self.Layer64x64((x128 + x64) / 2)
        
        x32 = self.FromRGB32x32(Input32x32)
        x32 = self.Layer32x32((x64 + x32) / 2)

        x16 = self.FromRGB16x16(Input16x16)
        x16 = self.Layer16x16((x32 + x16) / 2)
        
        x8 = self.FromRGB8x8(Input8x8)
        x8 = self.Layer8x8((x16 + x8) / 2)
        
        x4 = self.FromRGB4x4(Input4x4)
        x4 = self.Layer4x4((x8 + x4) / 2)
        
        return self.CriticLayer(nn.functional.leaky_relu(x4, 0.2, inplace=True)).squeeze()