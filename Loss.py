import torch
import torch.nn as nn

def R1GradientPenalty(RealSamples, Critics):
    Gradient4x4, Gradient8x8, Gradient16x16, Gradient32x32, Gradient64x64, Gradient128x128 = torch.autograd.grad(outputs=Critics.sum(), inputs=RealSamples, create_graph=True, only_inputs=True)
    
    GradientNorm4x4 = Gradient4x4.square().sum([1,2,3])
    GradientNorm8x8 = Gradient8x8.square().sum([1,2,3])
    GradientNorm16x16 = Gradient16x16.square().sum([1,2,3])
    GradientNorm32x32 = Gradient32x32.square().sum([1,2,3])
    GradientNorm64x64 = Gradient64x64.square().sum([1,2,3])
    GradientNorm128x128 = Gradient128x128.square().sum([1,2,3])
    
    MultiscaleGradientNorm = (GradientNorm4x4 + GradientNorm8x8 + GradientNorm16x16 + GradientNorm32x32 + GradientNorm64x64 + GradientNorm128x128) / len(RealSamples)
    
    return 0.5 * MultiscaleGradientNorm.mean()

def RelativisticHingeDiscriminatorLoss(RealSampleCritics, FakeSampleCritics):
    x = nn.functional.relu(1 - (RealSampleCritics - torch.mean(FakeSampleCritics)))
    y = nn.functional.relu(1 + (FakeSampleCritics - torch.mean(RealSampleCritics)))
    return (torch.mean(x) + torch.mean(y)) / 2

def RelativisticHingeGeneratorLoss(RealSampleCritics, FakeSampleCritics):
    x = nn.functional.relu(1 + (RealSampleCritics - torch.mean(FakeSampleCritics)))
    y = nn.functional.relu(1 - (FakeSampleCritics - torch.mean(RealSampleCritics)))
    return (torch.mean(x) + torch.mean(y)) / 2

def PathLengthRegularization(FakeSamples, Latent, MeanPathLength, Decay=0.01):
    FakeSample4x4, FakeSample8x8, FakeSample16x16, FakeSample32x32, FakeSample64x64, FakeSample128x128 = FakeSamples

    PerturbedImage4x4 = FakeSample4x4 * torch.randn_like(FakeSample4x4) / 4
    PerturbedImage8x8 = FakeSample8x8 * torch.randn_like(FakeSample8x8) / 8
    PerturbedImage16x16 = FakeSample16x16 * torch.randn_like(FakeSample16x16) / 16
    PerturbedImage32x32 = FakeSample32x32 * torch.randn_like(FakeSample32x32) / 32
    PerturbedImage64x64 = FakeSample64x64 * torch.randn_like(FakeSample64x64) / 64
    PerturbedImage128x128 = FakeSample128x128 * torch.randn_like(FakeSample128x128) / 128
    
    Jacobian, = torch.autograd.grad(outputs=[PerturbedImage4x4.sum(), PerturbedImage8x8.sum(), PerturbedImage16x16.sum(), PerturbedImage32x32.sum(), PerturbedImage64x64.sum(), PerturbedImage128x128.sum()], inputs=Latent, create_graph=True, only_inputs=True)
    PathLength = torch.sqrt((Jacobian / len(FakeSamples)).square().sum(1))
    AccumulatedMeanPathLength = MeanPathLength + Decay * (PathLength.mean() - MeanPathLength)
    PathLengthPenalty = (PathLength - AccumulatedMeanPathLength).square().mean()
    
    return PathLengthPenalty, AccumulatedMeanPathLength.detach()