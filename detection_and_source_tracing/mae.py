from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoImageProcessor, VideoMAEForVideoClassification, get_linear_schedule_with_warmup

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="transformers.feature_extraction_utils",
)
np.random.seed(0)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from utils import mydataset


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
    parser.add_argument(
        "--batch_size",
        required=False,
        type=int,
        default=4,
    )
    parser.add_argument(
        "--device",
        required=False,
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--hf_cache_dir",
        required=False,
        type=str,
        default=None,
    )

    return parser.parse_args()

def create_dataloader(df, processor, batch_size):
    dataset = mydataset.MAEDataset(
        videos_file=df["video_path"],
        labels=df["labels"],
        processor=processor,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        drop_last=False,
    )

def build_dataframe(args):
    video_paths = []
    labels = []

    if args.task == "source_tracing":
        if not args.fake_videos_path:
            raise ValueError("Please assign at least one fake videos path.")
        if args.label_number != len(args.fake_videos_path):
            raise ValueError("label_number must match the number of fake video paths.")
        for label, path in enumerate(args.fake_videos_path):
            files = find_video_files(path)
            video_paths.extend(files)
            labels.extend([label] * len(files))
    elif args.task == "detection":
        if args.label_number != 2:
            raise ValueError("For the detection task, label_number must be 2.")
        if not args.real_videos_path or not args.fake_videos_path:
            raise ValueError("Please assign both real_videos_path and fake_videos_path.")
        for path in args.real_videos_path:
            files = find_video_files(path)
            video_paths.extend(files)
            labels.extend([0] * len(files))
        for path in args.fake_videos_path:
            files = find_video_files(path)
            video_paths.extend(files)
            labels.extend([1] * len(files))
    else:
        raise ValueError(f"Unsupported task: {args.task}")

    if not video_paths:
        raise ValueError("No .mp4 files were found under the provided paths.")

    return pd.DataFrame({"video_path": video_paths, "labels": labels})


def find_video_files(directory):
    video_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".mp4"):
                full_path = os.path.join(root, file)
                video_files.append(full_path)
    return video_files

def train_model(model, data_loader, loss_fn, optimizer, scheduler, n_examples, device):
    model = model.train()
    losses = []
    correct_predictions = 0

    for d in tqdm(data_loader, desc="Training", leave=False):
        input_vids = d["input"].to(device)
        label = d["label"].to(device)
        input_video = input_vids.squeeze(1)
        output = model(pixel_values=input_video)
        preds = output.logits.argmax(dim=1)
        loss = loss_fn(output.logits, label)
        
        correct_predictions += torch.sum(preds == label)
        losses.append(loss.item())
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
    return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, n_examples, device):
    model = model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        all_preds = []
        all_labels = []
        for d in data_loader:
            input_vids = d["input"].to(device)
            label = d["label"].to(device)
            input_video = input_vids.squeeze(1)
            output = model(pixel_values=input_video)
            preds = output.logits.argmax(dim=1)
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

def hf_kwargs(args):
    if args.hf_cache_dir:
        return {"cache_dir": args.hf_cache_dir}
    return {}

def main(args):
    device = torch.device(args.device)
    if args.train == "True":
        print("load data...")
        df_data = build_dataframe(args)
        processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base", **hf_kwargs(args))
        model = VideoMAEForVideoClassification.from_pretrained(
            "MCG-NJU/videomae-base",
            num_labels=args.label_number,
            **hf_kwargs(args),
        )
        video_cls = model
        video_cls = video_cls.to(device)
        df_train, df_val = train_test_split(
            df_data,
            test_size=0.2,
            random_state=2024,
            stratify=df_data["labels"],
        )
        df_train = df_train.reset_index(drop=True)
        df_val = df_val.reset_index(drop=True)
        train_data_loader = create_dataloader(df_train, processor, args.batch_size)
        val_data_loader = create_dataloader(df_val, processor, args.batch_size)

        EPOCHS = args.epoch

        LR = args.learning_rate

        optimizer = AdamW(video_cls.parameters(), lr=LR)
        total_steps = len(train_data_loader) * EPOCHS

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps,
        )

        loss_fn = torch.nn.CrossEntropyLoss()

        for epoch in tqdm(range(EPOCHS), desc="Epochs"):
            print(f"Epoch {epoch + 1}/{EPOCHS}")
            print("-" * 10)
            
            train_acc, train_loss = train_model(
                video_cls,
                train_data_loader,
                loss_fn,
                optimizer,
                scheduler,
                len(train_data_loader.dataset),
                device,
            )
            print(f"Train Loss: {train_loss} ; Train Accuracy: {train_acc}")
            
            val_acc, val_loss = eval_model(
                video_cls,
                val_data_loader,
                loss_fn,
                len(val_data_loader.dataset),
                device,
            )
            print(f"Val Loss: {val_loss} ; Val Accuracy: {val_acc}")
        Path(args.save_checkpoint_dir).parent.mkdir(parents=True, exist_ok=True)
        torch.save(video_cls.state_dict(), args.save_checkpoint_dir)
    elif args.train == "False":
        print("load data...")
        if not args.load_pre_trained_model_state:
            raise ValueError("Please define --load_pre_trained_model_state.")
        df_data = build_dataframe(args)
        processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base", **hf_kwargs(args))
        model = VideoMAEForVideoClassification.from_pretrained(
            "MCG-NJU/videomae-base",
            num_labels=args.label_number,
            **hf_kwargs(args),
        )
        model.load_state_dict(torch.load(args.load_pre_trained_model_state, map_location="cpu"))
        model = model.to(device)
        print("load model...")
        val_data_loader = create_dataloader(df_data, processor, args.batch_size)
        loss_fn = torch.nn.CrossEntropyLoss()
        val_acc, val_loss = eval_model(model, val_data_loader, loss_fn, len(df_data), device)
        print(f"Val Loss: {val_loss} ; Val Accuracy: {val_acc}")

def cli():
    main(parse_args())

if __name__ == '__main__':
    cli()
