import argparse
import os
import torch
from torch.autograd import Variable
from tqdm import tqdm
import time
from models.deep_isp_model import DenoisingNet
from msr_demosaic import MSRDemosaic
from audio_dataset import AudioDataset
import deep_isp_utils as utils
from collections import OrderedDict
import shutil
import matplotlib.pyplot as plt
from loss import *
from datetime import datetime

import numpy as np
from torch import nn
import quantize
import actquant
import IPython.display as ipd
import pdb

transformation = utils.JointCompose([
    utils.JointHorizontalFlip(),
    utils.JointVerticalFlip(),
    #utils.JointNormailze(means = [0.485,0.456,0.406],stds = [1,1,1]), #TODO consider use
    utils.JointToTensor(),
])
val_transformation = utils.JointCompose([
    #utils.JointNormailze(means = [0.485,0.456,0.406],stds = [1,1,1]),
    utils.JointToTensor(),
])

num_denoise_layers = 20
quant = False
inject_noise = False
quant_bitwidth = 32
quant_epoch_step = 50
inject_act_noise = False
act_bitwidth = 32
act_quant = False
quant_start_stage = 0
weight_relu = False
weight_grad_after_quant = False
random_inject_noise = False
step = 19
num_workers = 1
wrpn = False

gpus = [0]

def load_model(model,checkpoint):

    new_state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        name = k[7:] if k[0:6] == 'module.' else k # remove `module. if needed (happen when the model created with DataParallel
        #new_state_dict[name] = v
        new_state_dict[name] = v if v.dim() > 1 or 'num_batches_tracked' in name else v*v.new_ones(1)

    # load params
    model.load_state_dict(new_state_dict, strict=False) #strict false in case the loaded doesn't have alll variables like running mean

model = DenoisingNet(in_channels=1, num_denoise_layers=num_denoise_layers, quant=quant , noise=inject_noise, bitwidth=quant_bitwidth, quant_epoch_step=quant_epoch_step,
                         act_noise=inject_act_noise , act_bitwidth= act_bitwidth , act_quant=act_quant, use_cuda=(gpus is not None), quant_start_stage=quant_start_stage,
                         weight_relu=weight_relu, weight_grad_after_quant=weight_grad_after_quant, random_inject_noise = random_inject_noise
                         , step=step, wrpn=wrpn)
model.cuda()
device = 'cuda:' + str(0)
torch.cuda.set_device(0)

checkpoint_file = "/home/tomhirshberg/project/Denoising-drone-rotors/output//2019-05-24_15-27-48/checkpoint.pth.tar" # checkpoint location
if os.path.isfile(checkpoint_file):
    print("loading checkpoint {}".format(checkpoint_file))
    checkpoint = torch.load(checkpoint_file, map_location=device)
    load_model(model, checkpoint)
else:
    print("can't load checkpoint file")
    exit()
# Load dataset

datapath = '/home/tomhirshberg/project/Denoising-drone-rotors/preprocess_audio/data.h5'
# testset = MSRDemosaic(root=datapath, train=False, transform=val_transformation)
# test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=num_workers)

# use train set to check overfitting
testset = AudioDataset(data_h5_path=datapath, add_rpm=False, train=False)
test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=num_workers)


# testset = AudioDataset(data_h5_path=datapath, train=False)
# test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=num_workers)

import IPython.display as ipd
from preprocess_audio.preprocess_audio import *
# convert back to audio
from preprocess_audio.postprocess_audio import *

# ipd.Audio('/home/simon/denoise/dataset/audio/file_example_WAV_1MG.wav')
# ipd.Audio(x, rate=sr) # load a NumPy array

def unbias_image(img):
    return  torch.clamp(img, 0 , 1.).data.squeeze(0).cpu().numpy()#.transpose(1, 2, 0) + 0.5  #the clamp is becuase the value should be between 0-1


# Run inference
loader = test_loader
N_CHANNELS = 513
N_FFT = 1024
fs = 52735
n = 5


# librosa.output.write_wav("/home/simon/denoise/dataset/audio/file_example_WAV_1MG_re.wav", x, fs)
# ipd.Audio(x, rate=fs) # load a NumPy array
def to_image(data):
    return data.data.cpu().squeeze(0).squeeze(0).numpy()

for batch_idx, (data, target, fname) in enumerate(tqdm(loader)):
    # display noisy sample
    plt.figure()
    print("data source")
    #pdb.set_trace()
    data_image = to_image(data)
    print("data shape:", data_image.shape)
    print(data_image.max())
    #pdb.set_trace()
    plt.imsave('/home/tomhirshberg/project/Denoising-drone-rotors/data/dat/data_{}.png'.format(batch_idx),data_image)
    y = spectogram_to_wav(data_image, N_CHANNELS, N_FFT, fs, dst_path='/home/tomhirshberg/project/Denoising-drone-rotors/data/dat/orig_{}.wav'.format(batch_idx))
    ipd.display(ipd.Audio(y, rate=fs)) # load a NumPy array
    
    # display target sample
    print("target")
    #pdb.set_trace()
    target_image = to_image(target)
    print("max: ", target_image.max())
    print("min: ", target_image.min())
    plt.imsave('/home/tomhirshberg/project/Denoising-drone-rotors/data/dat/target_{}.png'.format(batch_idx),target_image)
    t = spectogram_to_wav(target_image, N_CHANNELS, N_FFT, fs, dst_path='/home/tomhirshberg/project/Denoising-drone-rotors/data/dat/target_{}.wav'.format(batch_idx))
    ipd.display(ipd.Audio(t, rate=fs)) # load a NumPy array

    # infer noisy sample
    if gpus is not None:
        data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)

    with torch.no_grad():
        data, target = Variable(data), Variable(target)
    
    
     
    output = model(data)
    #pdb.set_trace()
    #output *= 1.0 / output.max()
    #output = torch.sigmoid(output)
    np_output = unbias_image(output).squeeze(0)
    #pdb.set_trace()
    np_output = to_image(output)
    # display infered sample
    print("output")
    print(np_output.max())
    print(np_output.min())
    #pdb.set_trace()
    plt.imsave('/home/tomhirshberg/project/Denoising-drone-rotors/data/dat/output_{}.png'.format(batch_idx),np_output)
    #pdb.set_trace()
    x = spectogram_to_wav(np_output, N_CHANNELS, N_FFT, fs, dst_path='/home/tomhirshberg/project/Denoising-drone-rotors/data/dat/test_{}.wav'.format(batch_idx))
    ipd.display(ipd.Audio(x, rate=fs))
