import os
import random
import numpy as np

import torch

from grad.utils import fix_len_compatibility
from grad_extend.utils import parse_filelist


class TextMelSpeakerDataset(torch.utils.data.Dataset):
    def __init__(self, filelist_path):
        super().__init__()
        self.filelist = parse_filelist(filelist_path, split_char='|')
        self._filter()
        print(f'----------{len(self.filelist)}----------')

    def _filter(self):
        items_new = []
        # segment = 200
        items_min = 250  # 10ms * 250 = 2.5 S
        items_max = 500  # 10ms * 400 = 5.0 S
        for mel, vec, pit, spk in self.filelist:
            if not os.path.isfile(mel):
                continue
            if not os.path.isfile(vec):
                continue
            if not os.path.isfile(pit):
                continue
            if not os.path.isfile(spk):
                continue
            temp = np.load(pit)
            usel = int(temp.shape[0] - 1)  # useful length
            if (usel < items_min):
                continue
            if (usel >= items_max):
                usel = items_max
            items_new.append([mel, vec, pit, spk, usel])
        self.filelist = items_new

    def get_triplet(self, item):
        # print(item)
        mel = item[0]
        vec = item[1]
        pit = item[2]
        spk = item[3]
        use = item[4]

        mel = torch.load(mel)
        vec = np.load(vec)
        vec = np.repeat(vec, 2, 0)  # 320 VEC -> 160 * 2
        pit = np.load(pit)
        spk = np.load(spk)

        vec = torch.FloatTensor(vec)
        pit = torch.FloatTensor(pit)
        spk = torch.FloatTensor(spk)

        len_vec = vec.size()[0] - 2 # for safe
        len_pit = pit.size()[0]
        len_min = min(len_pit, len_vec)

        mel = mel[:, :len_min]
        vec = vec[:len_min, :]
        pit = pit[:len_min]

        if len_min > use:
            max_frame_start = vec.size(0) - use - 1
            frame_start = random.randint(0, max_frame_start)
            frame_end = frame_start + use

            mel = mel[:, frame_start:frame_end]
            vec = vec[frame_start:frame_end, :]
            pit = pit[frame_start:frame_end]
        # print(mel.shape)
        # print(vec.shape)
        # print(pit.shape)
        # print(spk.shape)
        return (mel, vec, pit, spk)

    def __getitem__(self, index):
        mel, vec, pit, spk = self.get_triplet(self.filelist[index])
        item = {'mel': mel, 'vec': vec, 'pit': pit, 'spk': spk}
        return item

    def __len__(self):
        return len(self.filelist)

    def sample_test_batch(self, size):
        idx = np.random.choice(range(len(self)), size=size, replace=False)
        test_batch = []
        for index in idx:
            test_batch.append(self.__getitem__(index))
        return test_batch


class TextMelSpeakerBatchCollate(object):
    # mel: [freq, length]
    # vec: [len, 256]
    # pit: [len]
    # spk: [256]
    def __call__(self, batch):
        B = len(batch)
        mel_max_length = max([item['mel'].shape[-1] for item in batch])
        max_length = fix_len_compatibility(mel_max_length)

        d_mel = batch[0]['mel'].shape[0]
        d_vec = batch[0]['vec'].shape[1]
        d_spk = batch[0]['spk'].shape[0]
        # print("d_mel", d_mel)
        # print("d_vec", d_vec)
        # print("d_spk", d_spk)
        mel = torch.zeros((B, d_mel, max_length), dtype=torch.float32)
        vec = torch.zeros((B, max_length, d_vec), dtype=torch.float32)
        pit = torch.zeros((B, max_length), dtype=torch.float32)
        spk = torch.zeros((B, d_spk), dtype=torch.float32)
        lengths = torch.LongTensor(B)

        for i, item in enumerate(batch):
            y_, x_, p_, s_ = item['mel'], item['vec'], item['pit'], item['spk']

            mel[i, :, :y_.shape[1]] = y_
            vec[i, :x_.shape[0], :] = x_
            pit[i, :p_.shape[0]] = p_
            spk[i] = s_

            lengths[i] = y_.shape[1]
        # print("lengths", lengths.shape)
        # print("vec", vec.shape)
        # print("pit", pit.shape)
        # print("spk", spk.shape)
        # print("mel", mel.shape)
        return {'lengths': lengths, 'vec': vec, 'pit': pit, 'spk': spk, 'mel': mel}
