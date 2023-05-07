'''
ref: https://github.com/TaoRuijie/ECAPA-TDNN
Wav dataset for training
'''

# Target domain dataset (CN-Celeb1)

import glob, numpy, os, random, soundfile, torch
from scipy import signal

class wav_dset_2(object):
    def __init__(self, cfg, **kwargs):
        self.train_path = cfg.train_path
        self.num_frames = cfg.num_frames
        # Load and configure augmentation files
        self.noisetypes = ['noise','speech','music']
        self.noisesnr = {'noise':[0,15],'speech':[13,20],'music':[5,15]}
        self.numnoise = {'noise':[1,1], 'speech':[3,8], 'music':[1,1]}
        self.noiselist = {}
        augment_files   = glob.glob(os.path.join(cfg.musan_path,'*/*/*/*.wav'))
        for file in augment_files:
            if file.split('/')[-4] not in self.noiselist:
                self.noiselist[file.split('/')[-4]] = []
            self.noiselist[file.split('/')[-4]].append(file)
        self.rir_files  = glob.glob(os.path.join(cfg.rir_path,'*/*/*.wav'))
        # Load data & labels
        self.data_list  = []
        self.data_label = []
        lines = open(cfg.train_list).read().splitlines()
        dictkeys = list(set([x.split()[0] for x in lines]))
        dictkeys.sort()
        dictkeys = { key : ii for ii, key in enumerate(dictkeys) }
        for index, line in enumerate(lines):
            speaker_label = dictkeys[line.split()[0]]
            file_name     = os.path.join(cfg.train_path, line.split()[1])
            self.data_label.append(speaker_label)
            self.data_list.append(file_name)

    def __getitem__(self, index):
        # Read the utterance and split into two random segments
        segments = self.wav_split(index)
        # Data Augmentation
        with torch.no_grad():
            aug_segments = torch.FloatTensor([])
            for i in range(0, segments.shape[0]):
                audio = segments[i,:]
                audio = numpy.stack([audio],axis=0)
                augtype = random.randint(0,5)
                if augtype == 0:   # Original
                    audio = audio
                elif augtype == 1: # Reverberation
                    audio = self.add_rev(audio)
                elif augtype == 2: # Babble
                    audio = self.add_noise(audio, 'speech')
                elif augtype == 3: # Music
                    audio = self.add_noise(audio, 'music')
                elif augtype == 4: # Noise
                    audio = self.add_noise(audio, 'noise')
                elif augtype == 5: # Television noise
                    audio = self.add_noise(audio, 'speech')
                    audio = self.add_noise(audio, 'music')
                audio = torch.FloatTensor(audio[0])
                audio = audio.unsqueeze(0)
                aug_segments = torch.cat((aug_segments, audio), 0)
        return aug_segments, self.data_label[index]

    def __len__(self):
        return len(self.data_list)

    def add_rev(self, audio):
        rir_file    = random.choice(self.rir_files)
        rir, sr     = soundfile.read(rir_file)
        rir         = numpy.expand_dims(rir.astype(numpy.float),0)
        rir         = rir / numpy.sqrt(numpy.sum(rir**2))
        return signal.convolve(audio, rir, mode='full')[:,:self.num_frames * 160 + 240]

    def add_noise(self, audio, noisecat):
        clean_db    = 10 * numpy.log10(numpy.mean(audio ** 2)+1e-4) 
        numnoise    = self.numnoise[noisecat]
        noiselist   = random.sample(self.noiselist[noisecat], random.randint(numnoise[0],numnoise[1]))
        noises = []
        for noise in noiselist:
            noiseaudio, sr = soundfile.read(noise)
            length = self.num_frames * 160 + 240
            if noiseaudio.shape[0] <= length:
                shortage = length - noiseaudio.shape[0]
                noiseaudio = numpy.pad(noiseaudio, (0, shortage), 'wrap')
            start_frame = numpy.int64(random.random()*(noiseaudio.shape[0]-length))
            noiseaudio = noiseaudio[start_frame:start_frame + length]
            noiseaudio = numpy.stack([noiseaudio],axis=0)
            noise_db = 10 * numpy.log10(numpy.mean(noiseaudio ** 2)+1e-4) 
            noisesnr   = random.uniform(self.noisesnr[noisecat][0],self.noisesnr[noisecat][1])
            noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)
        noise = numpy.sum(numpy.concatenate(noises,axis=0),axis=0,keepdims=True)
        return noise + audio
    
    def wav_split(self, index):
        audio, sr = soundfile.read(self.data_list[index])
        max_audio = self.num_frames * 160 + 240
        audiosize = audio.shape[0]
        randsize = audiosize - (max_audio*2)
        assert randsize >= 1
        startframe = random.sample(range(0, randsize), 2)
        startframe.sort()
        startframe[1] += max_audio
        startframe = numpy.array(startframe)
        # for more permutation
        numpy.random.shuffle(startframe)
        feats = []
        for asf in startframe:
            feats.append(audio[int(asf):int(asf)+max_audio])
        segments = numpy.stack(feats, axis=0)
        return segments