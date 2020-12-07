import torch
from torch.utils import data
import json
import os
import numpy as np
import soundfile as sf
import h5py
import random



def normalize_tensor_wav(wav_tensor, eps=1e-8, std=None):
    mean = wav_tensor.mean(-1, keepdim=True)
    if std is None:
        std = wav_tensor.std(-1, keepdim=True)
    return (wav_tensor - mean) / (std + eps)


class SATBDataset(data.Dataset):
    """Dataset class for WHAM source separation and speech enhancement tasks.

    Args:
        json_dir (str): The path to the directory containing the json files.
        task (str): One of ``'enh_single'``, ``'enh_both'``, ``'sep_clean'`` or
            ``'sep_noisy'``.

            * ``'enh_single'`` for single speaker speech enhancement.
            * ``'enh_both'`` for multi speaker speech enhancement.
            * ``'sep_clean'`` for two-speaker clean source separation.
            * ``'sep_noisy'`` for two-speaker noisy source separation.

        sample_rate (int, optional): The sampling rate of the wav files.
        segment (float, optional): Length of the segments used for training,
            in seconds. If None, use full utterances (e.g. for test).
        nondefault_nsrc (int, optional): Number of sources in the training
            targets.
            If None, defaults to one for enhancement tasks and two for
            separation tasks.
        normalize_audio (bool): If True then both sources and the mixture are
            normalized with the standard deviation of the mixture.

    References
        - "WHAM!: Extending Speech Separation to Noisy Environments",
        Wichern et al. 2019
    """

    dataset_name = "SATB"

    def __init__(
        self,
        hdf5_filepath,
        use_case=1,
        partition="train",
        sample_rate=22050,
        segment=0.25,
        nondefault_nsrc=None,
        normalize_audio=False,
    ):
        super(SATBDataset, self).__init__()

        # Task setting
        self.hdf5_filepath = hdf5_filepath
        self.sample_rate = sample_rate
        self.normalize_audio = normalize_audio
        self.seg_len = None if segment is None else int(segment * sample_rate)
        self.EPS = 1e-8
        self.partition = partition
        self.use_case = use_case
        self.n_src = 4


        self.dataset   = h5py.File(hdf5_filepath, "r")
        self.partPerSongDict = {}
        self.allParts = [key for key in self.dataset[self.partition].keys()]
        self.sources = ['soprano','tenor','bass','alto']
        self.songs = list(self.dataset[self.partition]['soprano1'].keys())
        
        for part in self.allParts:
            songs = self.dataset[self.partition][part]
            for song in songs:
                if song in self.partPerSongDict:
                    self.partPerSongDict[song].append(part)
                else:
                    self.partPerSongDict[song] = [part]

        # List numbers of singer per part for a given song
        self.partCountPerSongDict = {}
        for song in self.partPerSongDict.keys():
            parts = self.partPerSongDict[song]
            parts = [x[:-1] for x in parts]
            self.partCountPerSongDict[song] = {i:parts.count(i) for i in parts}

            # If a part is missing from the song, add its key and 0 as part count
            diff = list(set(self.sources) - set(self.partCountPerSongDict[song].keys()))
            if len(diff) != 0:
                for missing_part in diff:
                    self.partCountPerSongDict[song][missing_part] = 0
        # import pdb;pdb.set_trace()


    def __len__(self):
        return len(self.songs)

    def __getitem__(self, idx):
        """Gets a mixture/sources pair.
        Returns:
            mixture, vstack([source_arrays])
        """
        # Random start

        randsong = self.songs[idx]

        part_count = self.partCountPerSongDict[randsong]

            # Use-Case: At most one singer per part
        if (self.use_case==0):
            max_num_singer_per_part = 1
            randsources = random.sample(self.sources, random.randint(1,len(self.sources)))                   # Randomize source pick if at most one singer per part
        # Use-Case: Exactly one singer per part
        elif (self.use_case==1):
            max_num_singer_per_part = 1
            randsources = self.sources                                                                  # Take all sources + Set num singer = 1
        # Use-Case: At least one singer per part
        else:
            max_num_singer_per_part = 4
            randsources = self.sources        

        startspl = 0
        endspl   = 0

        while startspl == 0:
            # try:
            randpart = random.choice(self.sources) + '1'
            startspl = random.randint(0,len(self.dataset[self.partition][randpart][randsong]['raw_wav'])-self.seg_len) # This assume that all stems are the same length
            # except:
            #     continue


        endspl   = startspl+self.seg_len

        # Get Random Sources: 
        randsources_for_song = [] 
        zero_source_counter = 0 

        # import pdb;pdb.set_trace()

        for source in randsources:
            # If no singer in part, default it to one and fill array with zeros later
            if part_count[source] > 0:
                max_for_part = part_count[source] if part_count[source] < max_num_singer_per_part\
                 else max_num_singer_per_part
            else:
                max_for_part = 1 

            num_singer_per_part = random.randint(1,max_for_part)                      # Get random number of singer per part based on max_for_part
            singer_num = random.sample(range(1,max_for_part+1),num_singer_per_part)   # Get random part number for the number of singer based off max_for_part
            randsources_for_part = np.repeat(source,num_singer_per_part)              # Repeat the parts according to the number of singer per group
            randsources_for_part = ["{}{}".format(a_, b_) for a_, b_ in zip(randsources_for_part, singer_num)] # Concatenate strings for part name
            randsources_for_song+=randsources_for_part

        out_shape  = np.zeros((self.seg_len,))
        out_shapes = {'soprano':np.copy(out_shape),'alto':np.copy(out_shape),'tenor':np.copy(out_shape),\
        'bass':np.copy(out_shape), 'mix':np.copy(out_shape)}

        # import pdb;pdb.set_trace()

        # Retrieve the chunks and store them in output shapes 
        zero_source_counter = 0                                        
        for source in randsources_for_song:

            # Try to retrieve chunk. If part doesn't exist, create array of zeros instead
            try:
                source_chunk = self.dataset[self.partition][source][randsong]['raw_wav'][startspl:endspl]              # Retrieve part's chunk
            except:
                zero_source_counter += 1
                source_chunk = np.zeros(self.seg_len)

            out_shapes[source[:-1]] = np.add(out_shapes[source[:-1]],source_chunk)# Store chunk in output shapes
            out_shapes['mix'] = np.add(out_shapes['mix'],source_chunk)            # Add the chunk to the mix
        
        # Scale down all the group chunks based off number of sources per group
        scaler = len(randsources_for_song) - zero_source_counter
        out_shapes['soprano'] = (out_shapes['soprano']/scaler)
        out_shapes['alto']    = (out_shapes['alto']/scaler)
        out_shapes['tenor']   = (out_shapes['tenor']/scaler)
        out_shapes['bass']    = (out_shapes['bass']/scaler)
        out_shapes['mix'] = (out_shapes['mix']/scaler)

        out_sources = []
        out_sources.append(out_shapes['soprano'])
        out_sources.append(out_shapes['alto'])
        out_sources.append(out_shapes['tenor'])
        out_sources.append(out_shapes['bass'])

        out_sources = torch.from_numpy(np.vstack(out_sources))

        out_mix = torch.from_numpy(out_shapes['mix'])

        if self.normalize_audio:
            m_std = out_mix.std(-1, keepdim=True)
            out_mix = normalize_tensor_wav(out_mix, eps=self.EPS, std=m_std)
            sources = normalize_tensor_wav(sources, eps=self.EPS, std=m_std)
        return out_mix.float(), out_sources.float()


    def get_infos(self):
        """Get dataset infos (for publishing models).

        Returns:
            dict, dataset infos with keys `dataset`, `task` and `licences`.
        """
        infos = dict()
        infos["dataset"] = self.dataset_name
        infos["use_case"] = self.use_case
        return infos
