import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import av
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import os
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.feature_extraction_utils")
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from transformers import AutoProcessor, AutoModel, get_linear_schedule_with_warmup
import os
from tqdm import tqdm
# np.random.seed(0)
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
#os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'
import sys
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms
# import videotransforms


import numpy as np

# from charades_dataset import Charades as Dataset
class CreateDataset(torch.utils.data.Dataset):
    def __init__(self,videos_file,labels,processor):
        self.videos_file = videos_file
        self.labels = labels
        # print(self.labels[:7])
        self.processor = processor
        self.processed_video = []
        self.processed_label = []
        index = 0
        for video_file,label in zip(self.videos_file,self.labels):
            # print(index)
            container = av.open(video_file)
            # print(label)
            # print(container.streams.video[0].frames)
            indices = self.sample_frame_indices(clip_len=16, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
            video = self.read_video_pyav(container, indices)
            if video is None:
                continue
            # print(video.shape)
            if video.shape[0] != 16:
                continue
            index+=1
            print(index,flush=True)
            processed_video = self.processor(videos=list(video), return_tensors="pt")
            # print(processed_video['pixel_values'].shape)
            pro_video = processed_video['pixel_values']
            self.processed_video.append(pro_video)
            self.processed_label.append(label)
        # self.processed_video = self.processed_video[:2000]
        # self.processed_label = self.processed_label[:2000]
            # if index == 2000:
            #     break
        # self.processed_video = np.stack(self.processed_video)
        print(self.processed_label.count(0),flush=True)
        print(self.processed_label.count(1),flush=True)
        print(self.processed_label.count(2),flush=True)
        print(self.processed_label.count(3),flush=True)
        print(self.processed_label.count(4),flush=True)
        print(self.processed_label.count(5),flush=True)
    def __len__(self):
        return len(self.processed_label)

    def __getitem__(self,item):
        # print("------------------")
        # print(item)
        # print(self.videos_file[item])
        # print("------------------")
        
        # video_file = self.videos_file[item]
        # label = self.labels[item]
        # video = load_video(video_file)
        # processed_video = self.processor(videos=list(video), return_tensors="pt")
        # # print(processed_video['pixel_values'].shape)
        # processed_video['pixel_values'] = processed_video['pixel_values']
        # print(item)
        # print(self.processed_video[item].shape)
        # print(self.labels[item])
        return {
            'input':self.processed_video[item],
            'label': torch.tensor(self.processed_label[item])
        }


    def read_video_pyav(self,container, indices):
        '''
        Decode the video with PyAV decoder.
        Args:
            container (`av.container.input.InputContainer`): PyAV container.
            indices (`List[int]`): List of frame indices to decode.
        Returns:
            result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
        '''
        frames = []
        container.seek(0)
        start_index = indices[0]
        end_index = indices[-1]
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_index:
                break
            if i >= start_index and i in indices:
                frames.append(frame)
        if frames:
            return np.stack([x.to_ndarray(format="rgb24") for x in frames])
        else:
            return None


    def sample_frame_indices(self, clip_len, frame_sample_rate, seg_len):
        '''
        Sample the first number of frame indices from the video.
        Args:
            clip_len (`int`): Total number of frames to sample.
            frame_sample_rate (`int`): Sample every n-th frame.
            seg_len (`int`): Maximum allowed index of sample's last frame.
        Returns:
            indices (`List[int]`): List of sampled frame indices
        '''
        clip_len = min(clip_len, seg_len)

        indices = list(range(0, clip_len * frame_sample_rate, frame_sample_rate))

        return indices

def load_video(video_path, frame_count=16, resize_shape=(224, 224)):
    container = av.open(video_path)
    stream = container.streams.video[0]
    stream.thread_type = 'AUTO'  

    frames = []
    
    for frame in container.decode(stream):
        # img = frame.to_image()  
        # img_data = np.array(img) 
        img_data = frame.to_ndarray(format="rgb24")
        frames.append(img_data)

        if len(frames) == frame_count:  
            break
    # print(len(frames))
    container.close()
    if frames:
        video_data = np.stack(frames)
        video_data = torch.tensor(video_data)
        return video_data
    else:
        return None

def VideoDataLoader(df,processor,batch_size):
    ds = CreateDataset(videos_file = df['video_path'],
                        labels = df['labels'],
                        processor = processor)
    print("use dataloader",flush=True)
    return torch.utils.data.DataLoader(ds,batch_size=batch_size,num_workers = 0,drop_last=True)

class MaxPool3dSamePadding(nn.MaxPool3d):
    
    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        #print t,h,w
        out_t = np.ceil(float(t) / float(self.stride[0]))
        out_h = np.ceil(float(h) / float(self.stride[1]))
        out_w = np.ceil(float(w) / float(self.stride[2]))
        #print out_t, out_h, out_w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        #print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        #print x.size()
        #print pad
        x = F.pad(x, pad)
        return super(MaxPool3dSamePadding, self).forward(x)
    

class Unit3D(nn.Module):

    def __init__(self, in_channels,
                 output_channels,
                 kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1),
                 padding=0,
                 activation_fn=F.relu,
                 use_batch_norm=True,
                 use_bias=False,
                 name='unit_3d'):
        
        """Initializes Unit3D module."""
        super(Unit3D, self).__init__()
        
        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding
        
        self.conv3d = nn.Conv3d(in_channels=in_channels,
                                out_channels=self._output_channels,
                                kernel_size=self._kernel_shape,
                                stride=self._stride,
                                padding=0, # we always want padding to be 0 here. We will dynamically pad based on input size in forward function
                                bias=self._use_bias)
        
        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels, eps=0.001, momentum=0.01)

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

            
    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        #print t,h,w
        out_t = np.ceil(float(t) / float(self._stride[0]))
        out_h = np.ceil(float(h) / float(self._stride[1]))
        out_w = np.ceil(float(w) / float(self._stride[2]))
        #print out_t, out_h, out_w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        #print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        #print x.size()
        #print pad
        x = F.pad(x, pad)
        #print x.size()        

        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x



class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, name):
        super(InceptionModule, self).__init__()

        self.b0 = Unit3D(in_channels=in_channels, output_channels=out_channels[0], kernel_shape=[1, 1, 1], padding=0,
                         name=name+'/Branch_0/Conv3d_0a_1x1')
        self.b1a = Unit3D(in_channels=in_channels, output_channels=out_channels[1], kernel_shape=[1, 1, 1], padding=0,
                          name=name+'/Branch_1/Conv3d_0a_1x1')
        self.b1b = Unit3D(in_channels=out_channels[1], output_channels=out_channels[2], kernel_shape=[3, 3, 3],
                          name=name+'/Branch_1/Conv3d_0b_3x3')
        self.b2a = Unit3D(in_channels=in_channels, output_channels=out_channels[3], kernel_shape=[1, 1, 1], padding=0,
                          name=name+'/Branch_2/Conv3d_0a_1x1')
        self.b2b = Unit3D(in_channels=out_channels[3], output_channels=out_channels[4], kernel_shape=[3, 3, 3],
                          name=name+'/Branch_2/Conv3d_0b_3x3')
        self.b3a = MaxPool3dSamePadding(kernel_size=[3, 3, 3],
                                stride=(1, 1, 1), padding=0)
        self.b3b = Unit3D(in_channels=in_channels, output_channels=out_channels[5], kernel_shape=[1, 1, 1], padding=0,
                          name=name+'/Branch_3/Conv3d_0b_1x1')
        self.name = name

    def forward(self, x):    
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0,b1,b2,b3], dim=1)


class InceptionI3d(nn.Module):
    """Inception-v1 I3D architecture.
    The model is introduced in:
        Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
        Joao Carreira, Andrew Zisserman
        https://arxiv.org/pdf/1705.07750v1.pdf.
    See also the Inception architecture, introduced in:
        Going deeper with convolutions
        Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
        Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
        http://arxiv.org/pdf/1409.4842v1.pdf.
    """

    # Endpoints of the model in order. During construction, all the endpoints up
    # to a designated `final_endpoint` are returned in a dictionary as the
    # second return value.
    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'Mixed_3b',
        'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        'MaxPool3d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c',
        'Logits',
        'Predictions',
    )

    def __init__(self, num_classes=400, spatial_squeeze=True,
                 final_endpoint='Logits', name='inception_i3d', in_channels=3, dropout_keep_prob=0.5):
        """Initializes I3D model instance.
        Args:
          num_classes: The number of outputs in the logit layer (default 400, which
              matches the Kinetics dataset).
          spatial_squeeze: Whether to squeeze the spatial dimensions for the logits
              before returning (default True).
          final_endpoint: The model contains many possible endpoints.
              `final_endpoint` specifies the last endpoint for the model to be built
              up to. In addition to the output at `final_endpoint`, all the outputs
              at endpoints up to `final_endpoint` will also be returned, in a
              dictionary. `final_endpoint` must be one of
              InceptionI3d.VALID_ENDPOINTS (default 'Logits').
          name: A string (optional). The name of this module.
        Raises:
          ValueError: if `final_endpoint` is not recognized.
        """

        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

        super(InceptionI3d, self).__init__()
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint
        self.logits = None

        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % self._final_endpoint)

        self.end_points = {}
        end_point = 'Conv3d_1a_7x7'
        self.end_points[end_point] = Unit3D(in_channels=in_channels, output_channels=64, kernel_shape=[7, 7, 7],
                                            stride=(2, 2, 2), padding=(3,3,3),  name=name+end_point)
        if self._final_endpoint == end_point: return
        
        end_point = 'MaxPool3d_2a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return
        
        end_point = 'Conv3d_2b_1x1'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=64, kernel_shape=[1, 1, 1], padding=0,
                                       name=name+end_point)
        if self._final_endpoint == end_point: return
        
        end_point = 'Conv3d_2c_3x3'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=192, kernel_shape=[3, 3, 3], padding=1,
                                       name=name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_3a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return
        
        end_point = 'Mixed_3b'
        self.end_points[end_point] = InceptionModule(192, [64,96,128,16,32,32], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_3c'
        self.end_points[end_point] = InceptionModule(256, [128,128,192,32,96,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_4a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4b'
        self.end_points[end_point] = InceptionModule(128+192+96+64, [192,96,208,16,48,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4c'
        self.end_points[end_point] = InceptionModule(192+208+48+64, [160,112,224,24,64,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4d'
        self.end_points[end_point] = InceptionModule(160+224+64+64, [128,128,256,24,64,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4e'
        self.end_points[end_point] = InceptionModule(128+256+64+64, [112,144,288,32,64,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4f'
        self.end_points[end_point] = InceptionModule(112+288+64+64, [256,160,320,32,128,128], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_5a_2x2'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5b'
        self.end_points[end_point] = InceptionModule(256+320+128+128, [256,160,320,32,128,128], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5c'
        self.end_points[end_point] = InceptionModule(256+320+128+128, [384,192,384,48,128,128], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Logits'
        self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7],
                                     stride=(1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits = Unit3D(in_channels=384+384+128+128, output_channels=self._num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')

        self.build()


    def replace_logits(self, num_classes):
        self._num_classes = num_classes
        self.logits = Unit3D(in_channels=384+384+128+128, output_channels=self._num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')
        
    
    def build(self):
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])
        
    def forward(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x) # use _modules to work with dataparallel

        x = self.logits(self.dropout(self.avg_pool(x)))
        if self._spatial_squeeze:
            logits = x.squeeze(3).squeeze(3)
        # logits is batch X time X classes, which is what we want to work with
        return logits
        

    def extract_features(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)
        return self.avg_pool(x)

def find_video_files(directory):
    video_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".mp4"):
                full_path = os.path.join(root, file)
                video_files.append(full_path)
    return video_files

def train_model(model, data_loader, loss_fn, optimizer, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0
    for d in tqdm(data_loader, desc="Training", leave=False):
        # print(d['input'].shape)
        input_vids = d['input'].squeeze(1).permute(0,2,1,3,4).to("cuda:0")
        label = d['label'].to("cuda:0")
        # print(input_vids.shape)
        output = model(input_vids).squeeze()
        # print("get output....")
        # print(output.shape)
        _, preds = torch.max(output, dim = 1)
        loss = loss_fn(output, label)
        
        correct_predictions += torch.sum(preds == label)
        losses.append(loss.item())
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        # print("finish update....")
        
    return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        all_preds = []
        all_labels = []
        for d in data_loader:
            
            input_vids = d['input'].squeeze(1).permute(0,2,1,3,4).to("cuda:0")
            label = d['label'].to("cuda:0")
            # input_video = input_vids['pixel_values']
            # input_video = input_video.squeeze(1)
            output = model(input_vids).squeeze()
            # print(output.shape)
            _, preds = torch.max(output, dim = 1)
            loss = loss_fn(output, label)
            
            correct_predictions += torch.sum(preds == label)
            losses.append(loss.item())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
        total_loss = sum(losses) / len(losses)
        total_correct = correct_predictions.double() / len(data_loader.dataset)

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        print(classification_report(all_labels, all_preds))    
        return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

def run():
    # setup dataset
    if train :
        print("load data....")
        
        # label0_path = find_video_files("./invid/clip")[1000:2000]
        # label1_path = find_video_files("./i2vgen-xl/outputs/invid/i2v")
        # label0_path = find_video_files("./Hotshot-XL/outputs/webvid")
        # label1_path = find_video_files("./i2vgen-xl/outputs/webvid/i2v")
        # label2_path = find_video_files("./i2vgen-xl/outputs/webvid/t2v")
        # label3_path = find_video_files("./LaVie/res/base/webvid")
        # label4_path = find_video_files("./SEINE/results/webvid/i2v")
        # label5_path = find_video_files("./Show-1/outputs/webvid")
        # label6_path = find_video_files("./video_prevention/outputs/webvid/svd_xt")
        # label7_path = find_video_files("./VideoCrafter/results/webvid/i2v")
        # label8_path = find_video_files("./VideoCrafter/results/webvid/t2v")
        label0_path = find_video_files("./Hotshot-XL/outputs/invid")
        label1_path = find_video_files("./i2vgen-xl/outputs/invid/i2v")
        label2_path = find_video_files("./i2vgen-xl/outputs/invid/t2v")
        label3_path = find_video_files("./LaVie/res/base/invid")
        label4_path = find_video_files("./SEINE/results/invid/i2v")
        label5_path = find_video_files("./Show-1/outputs/invid")
        label6_path = find_video_files("./video_prevention/outputs/Invid/svd_xt")
        label7_path = find_video_files("./VideoCrafter/results/invid/i2v")
        label8_path = find_video_files("./VideoCrafter/results/invid/t2v")
        label0 = np.full(len(label0_path),0)
        label1 = np.full(len(label1_path),1)
        label2 = np.full(len(label2_path),2)
        label3 = np.full(len(label3_path),3)
        label4 = np.full(len(label4_path),4)
        label5 = np.full(len(label5_path),5)
        label6 = np.full(len(label6_path),6)
        label7 = np.full(len(label7_path),7)
        label8 = np.full(len(label8_path),8)

        print(len(label0))
        print(len(label1))
        print(len(label2))
        print(len(label3))
        print(len(label4))
        print(len(label5))
        print(len(label6))
        print(len(label7))
        print(len(label8))
        label0_path = np.array(label0_path)
        label1_path = np.array(label1_path)
        label2_path = np.array(label2_path)
        label3_path = np.array(label3_path)
        label4_path = np.array(label4_path)
        label5_path = np.array(label5_path)
        label6_path = np.array(label6_path)
        label7_path = np.array(label7_path)
        label8_path = np.array(label8_path)


        # labels = np.concatenate((label0,label1))
        
        # video_path = np.concatenate((label0_path,label1_path))

        labels = np.concatenate((label0,label1,label2,label3,label4,label5,label6,label7,label8))
        
        video_path = np.concatenate((label0_path,label1_path,label2_path,label3_path,label4_path,label5_path,label6_path,label7_path,label8_path))

        data={}
        data['video_path'] = video_path
        data['labels'] = labels

        
        processor = AutoProcessor.from_pretrained("microsoft/xclip-large-patch14", cache_dir="/scratch/trv3px/huggingface/hub")
        df_data = pd.DataFrame(data)
        df_train, df_val = train_test_split(df_data,test_size = 0.2, random_state = 2024, stratify=df_data['labels'])
        df_train = df_train.reset_index(drop=True)
        df_val = df_val.reset_index(drop=True)
        train_data_loader = VideoDataLoader(df_train,processor,4)
        val_data_loader = VideoDataLoader(df_val,processor,4)
        print("load model....",flush=True)
        EPOCHS = 20

        LR = 1e-5
        i3d = InceptionI3d(400, in_channels=3)
        i3d.load_state_dict(torch.load('models/rgb_imagenet.pt'))
        i3d.replace_logits(9)
        i3d = i3d.to("cuda:0")
        print("start training...",flush=True)
        optimizer = AdamW(i3d.parameters(), lr = LR)
        total_steps = len(train_data_loader) * EPOCHS

        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, 
                                                num_training_steps = total_steps)
        loss_fn = torch.nn.CrossEntropyLoss()

        for epoch in tqdm(range(EPOCHS), desc="Epochs"):
            print(f'Epoch {epoch + 1}/{EPOCHS}',flush=True)
            print('-' * 10,flush=True)
            
            train_acc, train_loss = train_model(i3d, train_data_loader, loss_fn, optimizer, scheduler, len(df_train))
            print(f'Train Loss: {train_loss} ; Train Accuracy: {train_acc}',flush=True)
            
            val_acc, val_loss = eval_model(i3d, val_data_loader, loss_fn, len(df_val))
            print(f'Val Loss: {val_loss} ; Val Accuracy: {val_acc}',flush=True)
        torch.save(i3d.state_dict(), f'invid_I3D_ST_best_model.pth')
    else:
        print("load data....")
        
        # label0_path = find_video_files("./invid/clip")[1000:2000]
        # label1_path = find_video_files("./video_prevention/outputs/Invid/svd_xt")
        # label0 = np.full(len(label0_path),0)
        # label1 = np.full(len(label1_path),1)
        # label0_path = np.array(label0_path)
        # label1_path = np.array(label1_path)
        # labels = np.concatenate((label0,label1))
        
        # video_path = np.concatenate((label0_path,label1_path))
        label0_path = find_video_files("./Hotshot-XL/outputs/webvid")
        label1_path = find_video_files("./i2vgen-xl/outputs/webvid/i2v")
        label2_path = find_video_files("./i2vgen-xl/outputs/webvid/t2v")
        label3_path = find_video_files("./LaVie/res/base/webvid")
        label4_path = find_video_files("./SEINE/results/webvid/i2v")
        label5_path = find_video_files("./Show-1/outputs/webvid")
        label6_path = find_video_files("./video_prevention/outputs/webvid/svd_xt")
        label7_path = find_video_files("./VideoCrafter/results/webvid/i2v")
        label8_path = find_video_files("./VideoCrafter/results/webvid/t2v")

        label0 = np.full(len(label0_path),0)
        label1 = np.full(len(label1_path),1)
        label2 = np.full(len(label2_path),2)
        label3 = np.full(len(label3_path),3)
        label4 = np.full(len(label4_path),4)
        label5 = np.full(len(label5_path),5)
        label6 = np.full(len(label6_path),6)
        label7 = np.full(len(label7_path),7)
        label8 = np.full(len(label8_path),8)

        print(len(label0))
        print(len(label1))
        print(len(label2))
        print(len(label3))
        print(len(label4))
        print(len(label5))
        print(len(label6))
        print(len(label7))
        print(len(label8))
        label0_path = np.array(label0_path)
        label1_path = np.array(label1_path)
        label2_path = np.array(label2_path)
        label3_path = np.array(label3_path)
        label4_path = np.array(label4_path)
        label5_path = np.array(label5_path)
        label6_path = np.array(label6_path)
        label7_path = np.array(label7_path)
        label8_path = np.array(label8_path)


        # labels = np.concatenate((label0,label1))
        
        # video_path = np.concatenate((label0_path,label1_path))

        labels = np.concatenate((label0,label1,label2,label3,label4,label5,label6,label7,label8))
        
        video_path = np.concatenate((label0_path,label1_path,label2_path,label3_path,label4_path,label5_path,label6_path,label7_path,label8_path))


        data={}
        data['video_path'] = video_path
        data['labels'] = labels

        
        processor = AutoProcessor.from_pretrained("microsoft/xclip-large-patch14", cache_dir="/scratch/trv3px/huggingface/hub")
        df_data = pd.DataFrame(data)
        # df_train, df_val = train_test_split(df_data,test_size = 0.2, random_state = 2024, stratify=df_data['labels'])
        # df_train = df_train.reset_index(drop=True)
        # df_val = df_val.reset_index(drop=True)
        # train_data_loader = VideoDataLoader(df_train,processor,4)
        val_data_loader = VideoDataLoader(df_data,processor,4)
        print("load model....",flush=True)
        EPOCHS = 1

        LR = 1e-5
        i3d = InceptionI3d(400, in_channels=3)
        i3d.replace_logits(9)
        i3d.load_state_dict(torch.load("./invid_I3D_ST_best_model.pth"))
        i3d = i3d.to("cuda:0")
        print("start training...",flush=True)
        optimizer = AdamW(i3d.parameters(), lr = LR)
        total_steps = len(val_data_loader) * EPOCHS

        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, 
                                                num_training_steps = total_steps)
        loss_fn = torch.nn.CrossEntropyLoss()
        val_acc, val_loss = eval_model(i3d, val_data_loader, loss_fn, len(val_data_loader.dataset))
        print(f'Val Loss: {val_loss} ; Val Accuracy: {val_acc}',flush=True)



if __name__ == '__main__':
    # need to add argparse
    train = False
    run()