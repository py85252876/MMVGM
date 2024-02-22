import av
import torch
import pandas as pd
import torch.nn as nn
import numpy as np 
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from torch.optim import AdamW
import warnings
from sklearn.metrics import classification_report
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.feature_extraction_utils")
from sklearn.model_selection import train_test_split
from transformers import AutoProcessor, XCLIPVisionModel, get_linear_schedule_with_warmup, AutoModel,VivitImageProcessor,VivitModel, AutoImageProcessor,VideoMAEModel,VideoMAEForVideoClassification
from tqdm import tqdm
np.random.seed(0)
import argparse

import os
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
a_folder_path = os.path.join(parent_dir, 'utils')
import sys
sys.path.append(a_folder_path)
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

def CreateDataLoader(df,processor,batch_size):
    ds = mydataset.MAEDataset(videos_file = df['video_path'],
                        labels = df['labels'],
                        processor = processor)
    return torch.utils.data.DataLoader(ds,batch_size=batch_size,num_workers = 0,drop_last=True)

def process_files():
    if args.task == "source_tracing":
        data = {}
        if args.label_number != len(args.fake_videos_path):
            print("The label numbers is not equal with fake videos path, Please check and rerun.")
            return 
        for i,path in enumerate(args.fake_videos_path):
            data[f"label{i}_path"] = np.array(find_video_files(path)) 
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
        input_vids = d['input'].to("cuda:0")
        label = d['label'].to("cuda:0")
        input_video = input_vids.squeeze(1)
        output = model(pixel_values = input_video)
        _, preds = torch.max(output.logits , dim = 1)
        loss = loss_fn(output.logits, label)
        
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
            input_vids = d['input'].to("cuda:0")
            label = d['label'].to("cuda:0")
            input_video = input_vids.squeeze(1)
            output = model(input_video)
            _, preds = torch.max(output.logits, dim = 1)
            loss = loss_fn(output.logits, label)
            correct_predictions += torch.sum(preds == label)
            losses.append(loss.item())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
        total_loss = sum(losses) / len(losses)
        total_correct = correct_predictions.double() / len(data_loader.dataset)

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        print(classification_report(all_labels, all_preds))
            
        return correct_predictions.double() / n_examples, np.mean(losses)


def main():
    if args.train == "True":
        print("load data...")
        data = process_files()
        new_data = {}
        new_data['video_path'] = data['video_path']
        new_data['labels'] = data['labels']
        processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
        model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base",num_labels = args.label_number)
        video_cls = model
        video_cls = video_cls.to("cuda:0")
        df_data = pd.DataFrame(new_data)
        df_train, df_val = train_test_split(df_data,test_size = 0.2, random_state = 2024, stratify=df_data['labels'])
        df_train = df_train.reset_index(drop=True)
        df_val = df_val.reset_index(drop=True)
        train_data_loader = CreateDataLoader(df_train,processor,4)
        val_data_loader = CreateDataLoader(df_val,processor,4)

        EPOCHS = args.epoch

        LR = args.learning_rate

        optimizer = AdamW(video_cls.parameters(), lr = LR)
        total_steps = len(train_data_loader) * EPOCHS

        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, 
                                                num_training_steps = total_steps)

        loss_fn = torch.nn.CrossEntropyLoss()

        for epoch in tqdm(range(EPOCHS), desc="Epochs"):
            print(f'Epoch {epoch + 1}/{EPOCHS}')
            print('-' * 10)
            
            train_acc, train_loss = train_model(video_cls, train_data_loader, loss_fn, optimizer, scheduler, len(df_train))
            print(f'Train Loss: {train_loss} ; Train Accuracy: {train_acc}')
            
            val_acc, val_loss = eval_model(video_cls, val_data_loader, loss_fn, len(df_val))
            print(f'Val Loss: {val_loss} ; Val Accuracy: {val_acc}')
        torch.save(video_cls.state_dict(), args.save_checkpoint_dir)
    elif args.train == "False":
        print("load data...")
        data = process_files()
        new_data = {}
        new_data['video_path'] = data['video_path']
        new_data['labels'] = data['labels']
        processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base", cache_dir="../huggingface/hub")
        model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base", cache_dir="../huggingface/hub",num_labels = args.label_number)
        model.load_state_dict(torch.load(args.load_pre_trained_model_state))
        model = model.to("cuda:0")
        print("load model...")
        df_data = pd.DataFrame(new_data)
        val_data_loader = CreateDataLoader(df_data,processor,4)
        loss_fn = torch.nn.CrossEntropyLoss()
        val_acc, val_loss = eval_model(model, val_data_loader, loss_fn, len(df_data))
        print(f'Val Loss: {val_loss} ; Val Accuracy: {val_acc}')
if __name__ == '__main__':
    args = parse_args()
    main()