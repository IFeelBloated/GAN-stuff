import math
import torch
import torch.nn as nn

CompressionFactor = 4
LeakyReluAlpha = 0.2
LeakyReluGain = nn.init.calculate_gain('leaky_relu', LeakyReluAlpha)

def MSRInitializer(Layer, ActivationGain=1):
    FanIn = Layer.weight.data.size(1) * Layer.weight.data[0][0].numel()
    
    Layer.weight.data.normal_(0,  ActivationGain / math.sqrt(FanIn))
    Layer.bias.data.zero_()
    
    return Layer

class GeneratorBlock(nn.Module):
      def __init__(self, InputChannels, OutputChannels, ReceptiveField=3):
          super(GeneratorBlock, self).__init__()
          
          CompressedChannels = InputChannels // CompressionFactor
          
          self.LinearLayer1 = MSRInitializer(nn.Conv2d(InputChannels, CompressedChannels * 4, kernel_size=ReceptiveField, stride=1, padding=(ReceptiveField - 1) // 2, bias=True), ActivationGain=LeakyReluGain)
          self.LinearLayer2 = MSRInitializer(nn.Conv2d(CompressedChannels, OutputChannels, kernel_size=ReceptiveField, stride=1, padding=(ReceptiveField - 1) // 2, bias=True), ActivationGain=0)
          
          if InputChannels != OutputChannels:
              self.ShortcutLayer = MSRInitializer(nn.Conv2d(InputChannels, OutputChannels, kernel_size=1, stride=1, padding=0, bias=True), ActivationGain=LeakyReluGain)

      def forward(self, x):
          y = nn.functional.leaky_relu(x, LeakyReluAlpha)
          y = self.LinearLayer1(y)
          
          y = nn.functional.pixel_shuffle(y, 2)
          
          y = nn.functional.leaky_relu(y, LeakyReluAlpha, inplace=True)
          y = self.LinearLayer2(y)

          if hasattr(self, 'ShortcutLayer'):
              x = self.ShortcutLayer(nn.functional.leaky_relu(x, LeakyReluAlpha))
          
          return nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False) + y

class DiscriminatorBlock(nn.Module):
      def __init__(self, InputChannels, OutputChannels, ReceptiveField=3):
          super(DiscriminatorBlock, self).__init__()
          
          CompressedChannels = OutputChannels // CompressionFactor
          
          self.LinearLayer1 = MSRInitializer(nn.Conv2d(InputChannels, CompressedChannels, kernel_size=ReceptiveField, stride=1, padding=(ReceptiveField - 1) // 2, bias=True), ActivationGain=LeakyReluGain)
          self.LinearLayer2 = MSRInitializer(nn.Conv2d(CompressedChannels * 4, OutputChannels, kernel_size=ReceptiveField, stride=1, padding=(ReceptiveField - 1) // 2, bias=True), ActivationGain=0)
          
          if InputChannels != OutputChannels:
              self.ShortcutLayer = MSRInitializer(nn.Conv2d(InputChannels, OutputChannels, kernel_size=1, stride=1, padding=0, bias=True), ActivationGain=LeakyReluGain)
          
      def forward(self, x):
          y = nn.functional.leaky_relu(x, LeakyReluAlpha)
          y = self.LinearLayer1(y)

          y = nn.functional.pixel_unshuffle(y, 2)
          
          y = nn.functional.leaky_relu(y, LeakyReluAlpha, inplace=True)
          y = self.LinearLayer2(y)
          
          x = nn.functional.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False, recompute_scale_factor=True)
          if hasattr(self, 'ShortcutLayer'):
              x = self.ShortcutLayer(nn.functional.leaky_relu(x, LeakyReluAlpha, inplace=True))

          return x + y
     
class GeneratorOpeningBlock(nn.Module):
      def __init__(self, LatentDimension, OutputChannels, ReceptiveField=3):
          super(GeneratorOpeningBlock, self).__init__()
          
          self.LatentDimension = LatentDimension
          
          self.LinearLayer1 = MSRInitializer(nn.Linear(LatentDimension, LatentDimension * 4 * 4), ActivationGain=LeakyReluGain)
          self.LinearLayer2 = MSRInitializer(nn.Conv2d(LatentDimension, OutputChannels, kernel_size=ReceptiveField, stride=1, padding=(ReceptiveField - 1) // 2, bias=True), ActivationGain=LeakyReluGain)
          
      def forward(self, w):
          y = self.LinearLayer1(w)
          
          y = nn.functional.pixel_shuffle(y.view(-1, self.LatentDimension * 4 * 4, 1, 1), 4)
          
          y = nn.functional.leaky_relu(y, LeakyReluAlpha, inplace=True)
          
          return self.LinearLayer2(y)
    
class DiscriminatorClosingBlock(nn.Module):
      def __init__(self, InputChannels, LatentDimension, ReceptiveField=3):
          super(DiscriminatorClosingBlock, self).__init__()
          
          self.LatentDimension = LatentDimension
          
          self.LinearLayer1 = MSRInitializer(nn.Conv2d(InputChannels, LatentDimension, kernel_size=ReceptiveField, stride=1, padding=(ReceptiveField - 1) // 2, bias=True), ActivationGain=LeakyReluGain)
          self.LinearLayer2 = MSRInitializer(nn.Linear(LatentDimension * 4 * 4, LatentDimension), ActivationGain=LeakyReluGain)
          
      def forward(self, x):
          y = nn.functional.leaky_relu(x, LeakyReluAlpha, inplace=True)
          y = self.LinearLayer1(y)
          
          y = nn.functional.pixel_unshuffle(y, 4).view(-1, self.LatentDimension * 4 * 4)
          
          y = nn.functional.leaky_relu(y, LeakyReluAlpha, inplace=True)
          
          return self.LinearLayer2(y)
        
class ToRGB(nn.Module):
      def __init__(self, InputChannels, ResidualComponent=False):
          super(ToRGB, self).__init__()
          
          self.LinearLayer = MSRInitializer(nn.Conv2d(InputChannels, 3, kernel_size=1, stride=1, padding=0, bias=True), ActivationGain=0 if ResidualComponent else 1)

      def forward(self, x):
          x = nn.functional.leaky_relu(x, LeakyReluAlpha)
          return self.LinearLayer(x)
      
class MappingBlock(nn.Module):
      def __init__(self, LatentDimension):
          super(MappingBlock, self).__init__()
          
          self.LinearLayer1 = MSRInitializer(nn.Linear(LatentDimension, LatentDimension), ActivationGain=LeakyReluGain)
          self.LinearLayer2 = MSRInitializer(nn.Linear(LatentDimension, LatentDimension), ActivationGain=LeakyReluGain)
          
      def forward(self, z):
          y = z / torch.sqrt(torch.mean(z**2, dim=1, keepdim=True) + 1e-8) # spherical latent space
          y = self.LinearLayer1(y)
          y = nn.functional.leaky_relu(y, LeakyReluAlpha, inplace=True)
          y = self.LinearLayer2(y)
          return nn.functional.leaky_relu(y, LeakyReluAlpha, inplace=True)

class Generator(nn.Module):
    def __init__(self, LatentDimension):
        super(Generator, self).__init__()
        
        self.LatentLayer = MappingBlock(LatentDimension)
        
        self.Layer4x4 = GeneratorOpeningBlock(LatentDimension, 512 * 2)
        self.ToRGB4x4 = ToRGB(512 * 2)
        
        self.Layer8x8 = GeneratorBlock(512 * 2, 512 * 2)
        self.ToRGB8x8 = ToRGB(512 * 2, ResidualComponent=True)
        
        self.Layer16x16 = GeneratorBlock(512 * 2, 512)
        self.ToRGB16x16 = ToRGB(512, ResidualComponent=True)
        
        self.Layer32x32 = GeneratorBlock(512, 512)
        self.ToRGB32x32 = ToRGB(512, ResidualComponent=True)
        
        self.Layer64x64 = GeneratorBlock(512, 512)
        self.ToRGB64x64 = ToRGB(512, ResidualComponent=True)
        
        self.Layer128x128 = GeneratorBlock(512, 256)
        self.ToRGB128x128 = ToRGB(256, ResidualComponent=True)
      
    def forward(self, z):
        w = self.LatentLayer(z)
        
        y = self.Layer4x4(w)
        Output4x4 = self.ToRGB4x4(y)

        y = self.Layer8x8(y)
        Output8x8 = nn.functional.interpolate(Output4x4, scale_factor=2, mode='bilinear', align_corners=False) + self.ToRGB8x8(y)

        y = self.Layer16x16(y)
        Output16x16 = nn.functional.interpolate(Output8x8, scale_factor=2, mode='bilinear', align_corners=False) + self.ToRGB16x16(y)

        y = self.Layer32x32(y)
        Output32x32 = nn.functional.interpolate(Output16x16, scale_factor=2, mode='bilinear', align_corners=False) + self.ToRGB32x32(y)

        y = self.Layer64x64(y)
        Output64x64 = nn.functional.interpolate(Output32x32, scale_factor=2, mode='bilinear', align_corners=False) + self.ToRGB64x64(y)

        y = self.Layer128x128(y)
        Output128x128 = nn.functional.interpolate(Output64x64, scale_factor=2, mode='bilinear', align_corners=False) + self.ToRGB128x128(y)

        return w, torch.tanh(Output128x128)

class Discriminator(nn.Module):
    def __init__(self, LatentDimension):
        super(Discriminator, self).__init__()
        
        self.FromRGB = MSRInitializer(nn.Conv2d(3, 256, kernel_size=3, stride=1, padding=1, bias=True), ActivationGain=LeakyReluGain)
        
        self.Layer128x128 = DiscriminatorBlock(256, 512)
        self.Layer64x64 = DiscriminatorBlock(512, 512)
        self.Layer32x32 = DiscriminatorBlock(512, 512)
        self.Layer16x16 = DiscriminatorBlock(512, 512 * 2)
        self.Layer8x8 = DiscriminatorBlock(512 * 2, 512 * 2)
        self.Layer4x4 = DiscriminatorClosingBlock(512 * 2, LatentDimension)
        
        self.CriticLayer = MSRInitializer(nn.Linear(LatentDimension, 1))
        
    def forward(self, x):
        x = self.Layer128x128(self.FromRGB(x))
        x = self.Layer64x64(x)
        x = self.Layer32x32(x)
        x = self.Layer16x16(x)
        x = self.Layer8x8(x)
        x = self.Layer4x4(x)
        
        return self.CriticLayer(nn.functional.leaky_relu(x, LeakyReluAlpha, inplace=True)).squeeze()