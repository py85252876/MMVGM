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
import numpy as np
import os
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
a_folder_path = os.path.join(parent_dir, 'utils')
sys.path.append(a_folder_path)
import models
import mydataset


def parse_args():
    parser = argparse.ArgumentParser(description="I3D")
    parser.add_argument(
        "--load_pre_trained_model_state", 
        required=False,
        type=str,
        default=None
    )
    parser.add_argument(
        '--real_videos_path', 
        nargs='+', help='<Required> Set flag', 
        required=False)

    parser.add_argument(
        '--fake_videos_path', 
        nargs='+', help='<Required> Set flag', 
        required=False)

    parser.add_argument(
        "--pre_trained_I3D_model", 
        required=False,
        type=str,
        default=None
    )

    parser.add_argument(
        "--task",
        default="detection",
        choices=["detection","source_tracing"],
    )

    parser.add_argument(
        "--train", 
        required=True,
        type=str,
        default=True
    )

    parser.add_argument(
        "--learning_rate", 
        required=False,
        type=float,
        default=1e-5
    )

    parser.add_argument(
        "--epoch", 
        required=False,
        type=int,
        default=20
    )

    parser.add_argument(
        "--label_number", 
        required=False,
        type=int,
        default=9
    )

    parser.add_argument(
        "--save_checkpoint_dir", 
        required=False,
        type=str,
        default="./checkpoints.pt"
    )

    return parser.parse_args()

def VideoDataLoader(df,processor,batch_size):
    ds = mydataset.I3Ddataset(videos_file = df['video_path'],
                        labels = df['labels'],
                        processor = processor)
    print("use dataloader",flush=True)
    return torch.utils.data.DataLoader(ds,batch_size=batch_size,num_workers = 0,drop_last=True)

def process_files():
    if args.task == "source_tracing":
        data = {}
        if args.label_number != len(args.fake_videos_path):
            print("The label numbers is not equal with fake videos path, Please check and rerun.")
            return 
        for i,path in enumerate(args.fake_videos_path):
            data[f"label{i}_path"] = np.array(find_video_files(path)) #need to change
            data[f"label{i}"] = np.full(len(data[f"label{i}_path"]),i)
        data["video_path"] = np.concatenate([data[f'label{i}_path'] for i in range(args.label_number)])
        data['labels'] = np.concatenate([data[f'label{i}'] for i in range(args.label_number)])
        return data
    elif args.task == "detection":
        data = {}
        if args.label_number != 2:
            print("For detection task the label number should be 2.")
        if len(args.real_videos_path) == 0 or len(args.fake_videos_path) == 0:
            print("Please assign the path for real/fake videos.")
            return
        for i,path in enumerate(args.real_videos_path):
            data[f"real_label{i}_path"] = np.array(find_video_files(path))
            data[f"real_label{i}"] = np.full(len(data[f"real_label{i}_path"]),0)
        data["real_video_path"] = np.concatenate([data[f'real_label{i}_path'] for i in range(len(args.real_videos_path))])
        data['real_labels'] = np.concatenate([data[f'real_label{i}'] for i in range(len(args.real_videos_path))])
        for i,path in enumerate(args.fake_videos_path):
            data[f"fake_label{i}_path"] = np.array(find_video_files(path))
            data[f"fake_label{i}"] = np.full(len(data[f"fake_label{i}_path"]),1)
        data["fake_video_path"] = np.concatenate([data[f'fake_label{i}_path'] for i in range(len(args.fake_videos_path))])
        data['fake_labels'] = np.concatenate([data[f'fake_label{i}'] for i in range(len(args.fake_videos_path))])
        data['video_path'] = np.concatenate((data["real_video_path"],data["fake_video_path"]))
        data['labels'] = np.concatenate((data['real_labels'],data['fake_labels']))
        return data
    else:
        print("The task is wrong.")
        return

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
        input_vids = d['input'].squeeze(1).permute(0,2,1,3,4).to("cuda:0")
        label = d['label'].to("cuda:0")
        output = model(input_vids).squeeze()
        _, preds = torch.max(output, dim = 1)
        loss = loss_fn(output, label)
        
        correct_predictions += torch.sum(preds == label)
        losses.append(loss.item())
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
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
            output = model(input_vids).squeeze()
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

def main():
    # setup dataset
    if args.train == "True":
        print("load data....")
        data = process_files()
        new_data = {}
        new_data['video_path'] = data['video_path']
        new_data['labels'] = data['labels']
        
        processor = AutoProcessor.from_pretrained("microsoft/xclip-large-patch14")
        df_data = pd.DataFrame(new_data)
        df_train, df_val = train_test_split(df_data,test_size = 0.2, random_state = 2024, stratify=df_data['labels'])
        df_train = df_train.reset_index(drop=True)
        df_val = df_val.reset_index(drop=True)
        train_data_loader = VideoDataLoader(df_train,processor,4)
        val_data_loader = VideoDataLoader(df_val,processor,4)
        print("load model....",flush=True)
        EPOCHS = args.epoch

        LR = args.learning_rate
        i3d = models.InceptionI3d(400, in_channels=3)
        i3d.load_state_dict(torch.load(args.pre_trained_I3D_model))
        i3d.replace_logits(args.label_number)
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
        torch.save(i3d.state_dict(), args.save_checkpoint_dir)
    else:
        if args.load_pre_trained_model_state == "":
            print("Please define the pre-train model.")
            return 
        data = process_files()
        new_data = {}
        new_data['video_path'] = data['video_path']
        new_data['labels'] = data['labels']
        processor = AutoProcessor.from_pretrained("microsoft/xclip-large-patch14")
        df_data = pd.DataFrame(new_data)
        val_data_loader = VideoDataLoader(df_data,processor,4)
        print("load model....",flush=True)
        i3d = models.InceptionI3d(400, in_channels=3)
        i3d.replace_logits(args.label_number)
        print(args.load_pre_trained_model_state,flush=True)
        i3d.load_state_dict(torch.load(args.load_pre_trained_model_state))
        i3d = i3d.to("cuda:0")
        print("start training...",flush=True)
        loss_fn = torch.nn.CrossEntropyLoss()
        val_acc, val_loss = eval_model(i3d, val_data_loader, loss_fn, len(val_data_loader.dataset))
        print(f'Val Loss: {val_loss} ; Val Accuracy: {val_acc}',flush=True)



if __name__ == '__main__':
    # need to add argparse
    args = parse_args()
    main()