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
# from huggingface_hub import hf_hub_download
from tqdm import tqdm
np.random.seed(0)
import argparse



class CreateDataset(torch.utils.data.Dataset):
    def __init__(self,videos_file,labels,processor):
        self.videos_file = videos_file
        self.labels = labels
        # print(self.labels[:7])
        self.processor = processor
        self.processed_video = []
        self.processed_label = []
        for video_file,label in zip(self.videos_file,self.labels):
            try:
                print(len(self.processed_video))
                container = av.open(video_file)
                indices = self.sample_frame_indices(clip_len=16, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
                video = self.read_video_pyav(container, indices)
                processed_video = self.processor(list(video), return_tensors="pt")
                # print(processed_video['pixel_values'].shape)
                pro_video = processed_video['pixel_values']
                if pro_video.shape[1]==16:
                    self.processed_video.append(pro_video)
                    self.processed_label.append(label)
            except av.error.InvalidDataError as e:
                print(f"wrong file {video_file}: {e}")
            except Exception as e:
                print(f"mistake {video_file}: {e}")
        # self.processed_video = np.stack(self.processed_video)
    def __len__(self):
        return len(self.processed_video)

    def __getitem__(self,item):
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
        return np.stack([x.to_ndarray(format="rgb24") for x in frames])


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

def CreateDataLoader(df,processor,batch_size):
    ds = CreateDataset(videos_file = df['video_path'],
                        labels = df['labels'],
                        processor = processor)
    return torch.utils.data.DataLoader(ds,batch_size=batch_size,num_workers = 0,drop_last=True)

class BinaryClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super(BinaryClassifier, self).__init__()
        
        # Layer 1
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5)
        )


        self.layer2 = nn.Sequential(
            nn.Linear(1024, 512),
            # nn.BatchNorm1d(512),
            nn.ReLU6(),
            nn.Dropout(0.5)
        )
        
        # Layer 2
        self.layer3 = nn.Sequential(
            nn.Linear(512, 256),
            # nn.BatchNorm1d(256),
            nn.ReLU6(),
            nn.Dropout(0.5)
        )
        
        # Layer 3
        self.layer4 = nn.Sequential(
            nn.Linear(256, 128),
            # nn.BatchNorm1d(128),
            nn.ReLU6(),
            nn.Dropout(0.5)
        )
        
        # Layer 4
        self.layer5 = nn.Sequential(
            nn.Linear(128, 64),
            # nn.BatchNorm1d(64),
            nn.ReLU6(),
            nn.Dropout(0.5)
        )

        self.layer6 = nn.Sequential(
            nn.Linear(64, 32),
            # nn.BatchNorm1d(32),
            nn.ReLU6(),
            nn.Dropout(0.5)
        )
        
        # Output layer
        self.out_layer = nn.Linear(32, num_classes)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return self.out_layer(x)

class VideoClassifier(torch.nn.Module):
    def __init__(self,xclip,binary_cla):
        super(VideoClassifier,self).__init__()
        self.video_features_extractor = xclip
        self.classifier = binary_cla

    def forward(self,input_video):
        video_emb = self.video_features_extractor(pixel_values = input_video)
        classifier_output = self.classifier(video_emb)
        return classifier_output


def find_video_files(directory):
    video_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".mp4"):
                full_path = os.path.join(root, file)
                video_files.append(full_path)
    return video_files

def parse_args():
    parser = argparse.ArgumentParser(description="MAE")
    parser.add_argument(
        "--load_pre_trained_model_state", 
        required=False,
        type=str,
        default=None
    )

    parser.add_argument(
        "--train", 
        required=True,
        type=bool,
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
        type=float,
        default=20
    )

    parser.add_argument(
        "--label_number", 
        required=False,
        type=float,
        default=9
    )

    parser.add_argument(
        "--save_checkpoint_dir", 
        required=False,
        type=str,
        default="./checkpoints.pt"
    )

    return parser.parse_args()

def train_model(model, data_loader, loss_fn, optimizer, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0

    for d in tqdm(data_loader, desc="Training", leave=False):
        input_vids = d['input'].to("cuda:0")
        label = d['label'].to("cuda:0")
        # input_video = input_vids['pixel_values']
        input_video = input_vids.squeeze(1)
        output = model(pixel_values = input_video)
        # print(output.logits)
        # # logits = output.logits
        _, preds = torch.max(output.logits , dim = 1)
        # print(preds)
        # print(label)
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
            # input_video = input_vids['pixel_values']
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
    if args.train:

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


        labels = np.concatenate((label0,label1,label2,label3,label4,label5,label6,label7,label8))
        
        video_path = np.concatenate((label0_path,label1_path,label2_path,label3_path,label4_path,label5_path,label6_path,label7_path,label8_path))
        print("load data...")
        processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
        model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base",num_labels = args.label_number)
        video_cls = model
        video_cls = video_cls.to("cuda:0")
        print("load model...")
        data={}
        data['video_path'] = video_path
        data['labels'] = labels

        df_data = pd.DataFrame(data)
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
    else:
        
        # label1_path = find_video_files("./LaVie/res/base/invid")
        # label0_path = find_video_files("./invid/clip")[1000:2000] 
        # label1 = np.full(len(label1_path),1)
        # label0 = np.full(len(label0_path),0)
        # print(len(label1_path))
        # print(len(label0_path))
        # label0_path = np.array(label0_path)
        # label1_path = np.array(label1_path)

        # labels = np.concatenate((label1,label0))
        
        # video_path = np.concatenate((label1_path,label0_path))


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

        labels = np.concatenate((label0,label1,label2,label3,label4,label5,label6,label7,label8))
        
        video_path = np.concatenate((label0_path,label1_path,label2_path,label3_path,label4_path,label5_path,label6_path,label7_path,label8_path))


        print("load data...")
        processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base", cache_dir="/scratch/trv3px/huggingface/hub")
        model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base", cache_dir="/scratch/trv3px/huggingface/hub",num_labels = args.label_number)
        model.load_state_dict(torch.load(args.load_pre_trained_model_state))
        model = model.to("cuda:0")
        print(model)
        print("load model...")
        data={}
        data['video_path'] = video_path
        data['labels'] = labels

        df_data = pd.DataFrame(data)
        val_data_loader = CreateDataLoader(df_data,processor,4)
        loss_fn = torch.nn.CrossEntropyLoss()
        val_acc, val_loss = eval_model(model, val_data_loader, loss_fn, len(df_data))
        print(f'Val Loss: {val_loss} ; Val Accuracy: {val_acc}')
if __name__ == '__main__':
    args = parse_args()
    main(args)