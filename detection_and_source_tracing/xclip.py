import av
import torch
import pandas as pd
import torch.nn as nn
import numpy as np 
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from torch.optim import AdamW
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.feature_extraction_utils")
from sklearn.model_selection import train_test_split
from transformers import AutoProcessor, XCLIPVisionModel, get_linear_schedule_with_warmup, AutoModel
# from huggingface_hub import hf_hub_download
from tqdm import tqdm
np.random.seed(0)



class CreateDataset(torch.utils.data.Dataset):
    def __init__(self,videos_file,labels,processor):
        self.videos_file = videos_file
        self.labels = labels
        # print(self.labels[:7])
        self.processor = processor
        self.processed_video = []
        self.processed_label = []
        for video_file,label in zip(self.videos_file,self.labels):
            container = av.open(video_file)
            indices = self.sample_frame_indices(clip_len=8, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
            video = self.read_video_pyav(container, indices)
            if video is None:
                continue
            processed_video = self.processor(videos=list(video), return_tensors="pt")
            # print(processed_video['pixel_values'].shape)
            pro_video = processed_video['pixel_values']
            # print(pro_video.shape)
            if pro_video.shape[1] == 8:
                self.processed_video.append(pro_video)
                self.processed_label.append(label)
        print(self.processed_label.count(8))
        print(self.processed_label.count(7))
        print(self.processed_label.count(6))
        print(self.processed_label.count(5))
        print(self.processed_label.count(4))
        print(self.processed_label.count(3))
        print(self.processed_label.count(2))
        print(self.processed_label.count(1))
        print(self.processed_label.count(0))
    def __len__(self):
        return len(self.processed_label)

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
        # start_index = indices[0]
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

def CreateDataLoader(df,processor,batch_size):
    ds = CreateDataset(videos_file = df['video_path'],
                        labels = df['labels'],
                        processor = processor)
    return torch.utils.data.DataLoader(ds,batch_size=batch_size,num_workers = 0,drop_last=True)

class BinaryClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=9):
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
        # self.mit = mit
        # for param in self.video_features_extractor.parameters():
        #     param.requires_grad = False

    def forward(self,input_video):
        # print(input_video.shape)
        batch_size, num_frames, num_channels, height, width = input_video.shape
        # input_video = input_video.reshape(-1, num_channels, height, width)
        # print(self.mit.mit(self.mit.visual_projection(self.video_features_extractor(input_video).last_hidden_state[1]).view(4,16,-1))[1].shape)
        # print(self.video_features_extractor(input_video).last_hidden_state[0, 0, :].shape)
        # print(self.video_features_extractor(input_video).pooler_output.shape)
        video_emb = self.video_features_extractor.get_video_features(pixel_values = input_video)
        # video_emb = video_emb
        # print(video_emb.shape)
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

def train_model(model, data_loader, loss_fn, optimizer, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0
    all_preds = []
    all_labels = []
    for d in tqdm(data_loader, desc="Training", leave=False):
        input_vids = d['input'].to("cuda:0")
        label = d['label'].to("cuda:0")
        # input_video = input_vids['pixel_values']
        input_video = input_vids.squeeze(1)
        # print(input_video.shape)
        # input_video = input_video
        # label = label.to("cuda:0")
        output = model(input_video)
        # print(output)
        # # logits = output.logits
        _, preds = torch.max(output, dim = 1)
        # print(preds)
        # print(label)
        loss = loss_fn(output, label)
        # print(output)
        correct_predictions += torch.sum(preds == label)
        losses.append(loss.item())

        
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(label.cpu().numpy())
    total_loss = sum(losses) / len(losses)
    total_correct = correct_predictions.double() / len(data_loader.dataset)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    print(classification_report(all_labels, all_preds),flush=True)
        
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
            
        return correct_predictions.double() / n_examples, np.mean(losses)


def main():
    # label1_path = find_video_files("/scratch/trv3px/video_classification/i2vgen-xl/outputs/invid/i2v")
    # label0_path = find_video_files("/scratch/trv3px/video_classification/invid/clip")[0:1000]
    # # label0_path.extend(find_video_files("/scratch/trv3px/video_classification/webvid/data/videos/000051_000100"))
    # # label0_path.extend(find_video_files("/scratch/trv3px/video_classification/webvid/data/videos/000101_000150"))
    # # label0_path.extend(find_video_files("/scratch/trv3px/video_classification/webvid/data/videos/000151_000200"))
    # # label0_path.extend(find_video_files("/scratch/trv3px/video_classification/webvid/data/videos/000201_000250"))
    # # label0_path.extend(find_video_files("/scratch/trv3px/video_classification/webvid/data/videos/000251_000300"))
    # # label0_path.extend(find_video_files("/scratch/trv3px/video_classification/webvid/data/videos/000301_000350"))
    # # label0_path.extend(find_video_files("/scratch/trv3px/video_classification/webvid/data/videos/000351_000400"))
    # # label0_path.extend(find_video_files("/scratch/trv3px/video_classification/webvid/data/videos/000401_000450"))
    # # label0_path.extend(find_video_files("/scratch/trv3px/video_classification/webvid/data/videos/000451_000500"))
    # # label0_path.extend(find_video_files("/scratch/trv3px/video_classification/webvid/data/videos/000501_000550"))
    # # label0_path.extend(find_video_files("/scratch/trv3px/video_classification/webvid/data/videos/000551_000600"))
    # # label0_path.extend(find_video_files("/scratch/trv3px/video_classification/webvid/data/videos/000601_000650"))
    # # label0_path.extend(find_video_files("/scratch/trv3px/video_classification/webvid/data/videos/000651_000700"))
    # # label0_path.extend(find_video_files("/scratch/trv3px/video_classification/webvid/data/videos/000701_000750"))
    # # label0_path.extend(find_video_files("/scratch/trv3px/video_classification/webvid/data/videos/000751_000800"))
    # # label0_path.extend(find_video_files("/scratch/trv3px/video_classification/webvid/data/videos/000801_000850"))
    # # label0_path.extend(find_video_files("/scratch/trv3px/video_classification/webvid/data/videos/000851_000900"))
    # # label0_path.extend(find_video_files("/scratch/trv3px/video_classification/webvid/data/videos/000901_000950"))
    # # label0_path.extend(find_video_files("/scratch/trv3px/video_classification/webvid/data/videos/000951_001000"))
    

    # label1 = np.full(len(label1_path),1)
    # label0 = np.full(len(label0_path),0)
    # print(len(label1_path))
    # print(len(label0_path))
    # # label0_path = np.array(label0_path)
    # # label1_path = np.array(label1_path)

    # labels = np.concatenate((label1,label0))
    
    # video_path = np.concatenate((label1_path,label0_path))
    if train:
        # print("start training..",flush = True)
        # label1_path = find_video_files("/scratch/trv3px/video_classification/SEINE/results/webvid/i2v")
        # label0_path = find_video_files("/scratch/trv3px/video_classification/webvid/data/videos/000001_000050")[0:1000]
        # label1 = np.full(len(label1_path),1)
        # label0 = np.full(len(label0_path),0)
        # print(len(label1_path))
        # print(len(label0_path))
        # labels = np.concatenate((label1,label0))
        # video_path = np.concatenate((label1_path,label0_path))
        label0_path = find_video_files("/scratch/trv3px/video_classification/Hotshot-XL/outputs/invid")
        label1_path = find_video_files("/scratch/trv3px/video_classification/i2vgen-xl/outputs/invid/i2v")
        label2_path = find_video_files("/scratch/trv3px/video_classification/i2vgen-xl/outputs/invid/t2v")
        label3_path = find_video_files("/scratch/trv3px/video_classification/LaVie/res/base/invid")
        label4_path = find_video_files("/scratch/trv3px/video_classification/SEINE/results/invid/i2v")
        label5_path = find_video_files("/scratch/trv3px/video_classification/Show-1/outputs/invid")
        label6_path = find_video_files("/scratch/trv3px/video_classification/video_prevention/outputs/Invid/svd_xt")
        label7_path = find_video_files("/scratch/trv3px/video_classification/VideoCrafter/results/invid/i2v")
        label8_path = find_video_files("/scratch/trv3px/video_classification/VideoCrafter/results/invid/t2v")
        
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
        # print(len(video_path))
        # return
        print("load data...")
        processor = AutoProcessor.from_pretrained("microsoft/xclip-large-patch14", cache_dir="/scratch/trv3px/huggingface/hub")
        xclip = AutoModel.from_pretrained("microsoft/xclip-large-patch14", cache_dir="/scratch/trv3px/huggingface/hub")
        # mit = AutoModel.from_pretrained("microsoft/xclip-large-patch14", cache_dir="/scratch/trv3px/huggingface/hub")
        # print(xclip.config)
        # print(mit.config)
        binary_cla = BinaryClassifier(768,num_classes = 9)
        video_cls = VideoClassifier(xclip,binary_cla)
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

        EPOCHS = 20

        LR = 1e-4

        optimizer = AdamW(video_cls.parameters(), lr = LR)
        total_steps = len(train_data_loader) * EPOCHS

        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 200, 
                                                num_training_steps = total_steps)

        loss_fn = torch.nn.CrossEntropyLoss()

        for epoch in tqdm(range(EPOCHS), desc="Epochs"):
            print(f'Epoch {epoch + 1}/{EPOCHS}')
            print('-' * 10)
            
            train_acc, train_loss = train_model(video_cls, train_data_loader, loss_fn, optimizer, scheduler, len(df_train))
            print(f'Train Loss: {train_loss} ; Train Accuracy: {train_acc}')
            
            val_acc, val_loss = eval_model(video_cls, val_data_loader, loss_fn, len(df_val))
            print(f'Val Loss: {val_loss} ; Val Accuracy: {val_acc}')
        torch.save(video_cls.state_dict(), f'invid_xclip_ST_best_model.pth')
    else:
        print("start eval..",flush=True)
        # label1_path = find_video_files("/scratch/trv3px/video_classification/VideoCrafter/results/invid/t2v")
        # label0_path = find_video_files("/scratch/trv3px/video_classification/invid/clip")[0:1000]
        # # label1_path = find_video_files("/scratch/trv3px/video_classification/i2vgen-xl/outputs/invid/t2v")
        # # label0_path = find_video_files("/scratch/trv3px/video_classification/invid/clip")[0:1000]
        # label1 = np.full(len(label1_path),1)
        # label0 = np.full(len(label0_path),0)
        # print(len(label1_path))
        # print(len(label0_path))
        # labels = np.concatenate((label1,label0))
        # video_path = np.concatenate((label1_path,label0_path))


        label0_path = find_video_files("/scratch/trv3px/video_classification/Hotshot-XL/outputs/webvid")
        label1_path = find_video_files("/scratch/trv3px/video_classification/i2vgen-xl/outputs/webvid/i2v")
        label2_path = find_video_files("/scratch/trv3px/video_classification/i2vgen-xl/outputs/webvid/t2v")
        label3_path = find_video_files("/scratch/trv3px/video_classification/LaVie/res/base/webvid")
        label4_path = find_video_files("/scratch/trv3px/video_classification/SEINE/results/webvid/i2v")
        label5_path = find_video_files("/scratch/trv3px/video_classification/Show-1/outputs/webvid")
        label6_path = find_video_files("/scratch/trv3px/video_classification/video_prevention/outputs/webvid/svd_xt")
        label7_path = find_video_files("/scratch/trv3px/video_classification/VideoCrafter/results/webvid/i2v")
        label8_path = find_video_files("/scratch/trv3px/video_classification/VideoCrafter/results/webvid/t2v")
        
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
        processor = AutoProcessor.from_pretrained("microsoft/xclip-large-patch14", cache_dir="/scratch/trv3px/huggingface/hub")
        xclip = AutoModel.from_pretrained("microsoft/xclip-large-patch14", cache_dir="/scratch/trv3px/huggingface/hub")
        binary_cla = BinaryClassifier(768,num_classes = 9)
        video_cls = VideoClassifier(xclip,binary_cla)
        video_cls.load_state_dict(torch.load("/scratch/trv3px/video_classification/VideoX/X-CLIP/invid_xclip_ST_best_model.pth"))
        video_cls = video_cls.to("cuda:0")
        print("load model...")
        data={}
        data['video_path'] = video_path
        data['labels'] = labels

        df_data = pd.DataFrame(data)
        val_data_loader = CreateDataLoader(df_data,processor,4)

        EPOCHS = 1

        LR = 1e-4

        optimizer = AdamW(video_cls.parameters(), lr = LR)
        # total_steps = len(train_data_loader) * EPOCHS

        # scheduler = get_linear_schedule_with_warmup(optimizer, 
        #                                         num_warmup_steps = 200, 
        #                                         num_training_steps = total_steps)

        loss_fn = torch.nn.CrossEntropyLoss()
        val_acc, val_loss = eval_model(video_cls, val_data_loader, loss_fn, len(val_data_loader.dataset))
        print(f'Val Loss: {val_loss} ; Val Accuracy: {val_acc}')

if __name__ == '__main__':
    train = False
    
    main()