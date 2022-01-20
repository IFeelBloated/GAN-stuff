import torch
import torch.nn as nn
import math

def R1GradientPenalty(RealSamples, Critics):
    Gradient, = torch.autograd.grad(outputs=Critics.sum(), inputs=RealSamples, create_graph=True, only_inputs=True)
    return 0.5 * Gradient.square().sum([1,2,3]).mean()

def RelativisticHingeDiscriminatorLoss(RealSampleCritics, FakeSampleCritics):
    x = nn.functional.relu(1 - (RealSampleCritics - torch.mean(FakeSampleCritics)))
    y = nn.functional.relu(1 + (FakeSampleCritics - torch.mean(RealSampleCritics)))
    return (torch.mean(x) + torch.mean(y)) / 2

def RelativisticHingeGeneratorLoss(RealSampleCritics, FakeSampleCritics):
    x = nn.functional.relu(1 + (RealSampleCritics - torch.mean(FakeSampleCritics)))
    y = nn.functional.relu(1 - (FakeSampleCritics - torch.mean(RealSampleCritics)))
    return (torch.mean(x) + torch.mean(y)) / 2

def PathLengthRegularization(FakeSamples, Latent, MeanPathLength, Decay=0.01):
    PerturbedSamples = FakeSamples * torch.randn_like(FakeSamples) / math.sqrt(FakeSamples.shape[2] * FakeSamples.shape[3])
    Jacobian, = torch.autograd.grad(outputs=PerturbedSamples.sum(), inputs=Latent, create_graph=True, only_inputs=True)
    PathLength = torch.sqrt(Jacobian.square().sum(1))
    AccumulatedMeanPathLength = MeanPathLength + Decay * (PathLength.mean() - MeanPathLength)
    PathLengthPenalty = (PathLength - AccumulatedMeanPathLength).square().mean()
    
    return PathLengthPenalty, AccumulatedMeanPathLength.detach()