import torch
import av
import numpy as np

class I3Ddataset(torch.utils.data.Dataset):
    def __init__(self,videos_file,labels,processor):
        self.videos_file = videos_file
        self.labels = labels
        self.processor = processor
        self.processed_video = []
        self.processed_label = []
        for video_file,label in zip(self.videos_file,self.labels):
            try:
                # print(len(self.processed_video))
                container = av.open(video_file)
                indices = self.sample_frame_indices(clip_len=16, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
                video = self.read_video_pyav(container, indices)
                if video is None:
                    continue
                if video.shape[0] != 16:
                    continue
                processed_video = self.processor(videos=list(video), return_tensors="pt")
                pro_video = processed_video['pixel_values']
                if pro_video.shape[1]==16:
                    self.processed_video.append(pro_video)
                    self.processed_label.append(label)
            except av.error.InvalidDataError as e:
                print(f"wrong file {video_file}: {e}")
            except Exception as e:
                print(f"mistake {video_file}: {e}")
    def __len__(self):
        return len(self.processed_video)

    def __getitem__(self,item):
        return {
            'input':self.processed_video[item],
            'label': torch.tensor(self.processed_label[item])
        }

    def read_video_pyav(self,container, indices):
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
        
        clip_len = min(clip_len, seg_len)

        indices = list(range(0, clip_len * frame_sample_rate, frame_sample_rate))

        return indices

class MAEDataset(torch.utils.data.Dataset):
    def __init__(self,videos_file,labels,processor):
        self.videos_file = videos_file
        self.labels = labels
        self.processor = processor
        self.processed_video = []
        self.processed_label = []
        for video_file,label in zip(self.videos_file,self.labels):
            try:
                container = av.open(video_file)
                indices = self.sample_frame_indices(clip_len=16, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
                video = self.read_video_pyav(container, indices)
                processed_video = self.processor(list(video), return_tensors="pt")
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


class XCLIPDataset(torch.utils.data.Dataset):
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