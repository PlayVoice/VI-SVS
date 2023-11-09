import os
import numpy as np
import torch
import torch.utils.data

from vits.utils import load_wav_to_torch
from vits.spectrogram import spectrogram_torch
from pitch.utils import fix_len_compatibility


def load_filepaths(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths = [line.strip().split(split) for line in f]
    return filepaths


class TextAudioLoader(torch.utils.data.Dataset):
    """
    1) loads audio, text pairs
    2) normalizes text and converts them to sequences of integers
    3) computes spectrograms from audio files.
    """

    def __init__(self, audiopaths_and_text, hparams):
        self.audiopaths_and_text = load_filepaths(audiopaths_and_text)
        self.max_wav_value  = hparams.max_wav_value
        self.sampling_rate  = hparams.sampling_rate
        self.filter_length  = hparams.filter_length 
        self.hop_length     = hparams.hop_length 
        self.win_length     = hparams.win_length
        self.sampling_rate  = hparams.sampling_rate 
        self.min_text_len   = getattr(hparams, "min_text_len", 1)
        self.max_text_len   = getattr(hparams, "max_text_len", 5000)
        self._filter()
        print(f"~~~~~~~~~~~~~~~~~~~~~{len(self)}~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length
        audiopaths_and_text_new = []
        lengths = []
        for audiopath, text, score, pitch, slur in self.audiopaths_and_text:
            if self.min_text_len <= len(text) and len(text) <= self.max_text_len:
                wav_len = os.path.getsize(audiopath) // (2 * self.hop_length)
                if wav_len < 50: # no use short wave
                    continue
                audiopaths_and_text_new.append([audiopath, text, score, pitch, slur])
                lengths.append(wav_len)
        self.audiopaths_and_text = audiopaths_and_text_new
        self.lengths = lengths

    def get_audio_text_pair(self, audiopath_and_text):
        # separate filename and text
        file = audiopath_and_text[0]
        phone = audiopath_and_text[1]
        score = audiopath_and_text[2]
        pitch = audiopath_and_text[3]
        slurs = audiopath_and_text[4]

        phone, score, pitch, slurs = self.get_labels(phone, score, pitch, slurs)
        spec, wav = self.get_audio(file)

        len_phone = phone.size()[0]
        len_spec = spec.size()[-1]

        if len_phone != len_spec:
            # print("**************CareFull*******************")
            # print(f"filepath={audiopath_and_text[0]}")
            # print(f"len_text={len_phone}")
            # print(f"len_spec={len_spec}")
            if len_phone > len_spec:
                print(file)
                print("len_phone", len_phone)
                print("len_spec", len_spec)
            assert len_phone < len_spec
            len_min = min(len_phone, len_spec)
            len_wav = len_min * self.hop_length
            # print(wav.size())
            # print(f"len_min={len_min}")
            # print(f"len_wav={len_wav}")
            spec = spec[:, :len_min]
            wav = wav[:, :len_wav]
        return (phone, score, pitch, slurs, spec, wav)

    def get_labels(self, phone, score, pitch, slurs):
        phone = np.load(phone)
        score = np.load(score)
        pitch = np.load(pitch)
        slurs = np.load(slurs)
        phone = torch.LongTensor(phone)
        score = torch.LongTensor(score)
        pitch = torch.FloatTensor(pitch)
        slurs = torch.LongTensor(slurs)
        return phone, score, pitch, slurs

    def get_audio(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError(
                "{} {} SR doesn't match target {} SR".format(
                    sampling_rate, self.sampling_rate
                )
            )
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec.pt")
        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename)
        else:
            spec = spectrogram_torch(
                audio_norm,
                self.filter_length,
                self.sampling_rate,
                self.hop_length,
                self.win_length,
                center=False,
            )
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename)
        return spec, audio_norm

    def __getitem__(self, index):
        return self.get_audio_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextAudioCollate:
    """Zero-pads model inputs and targets"""

    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text and aduio
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[4].size(1) for x in batch]), dim=0, descending=True
        )

        max_phone_len = max([len(x[0]) for x in batch])
        # For Unet
        max_phone_len = fix_len_compatibility(max_phone_len)

        phone_lengths = torch.LongTensor(len(batch))
        phone_padded = torch.LongTensor(len(batch), max_phone_len)
        score_padded = torch.LongTensor(len(batch), max_phone_len)
        pitch_padded = torch.FloatTensor(len(batch), max_phone_len)
        slurs_padded = torch.LongTensor(len(batch), max_phone_len)
        phone_padded.zero_()
        score_padded.zero_()
        pitch_padded.zero_()
        slurs_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            phone = row[0]
            phone_padded[i, : phone.size(0)] = phone
            phone_lengths[i] = phone.size(0)

            score = row[1]
            score_padded[i, : score.size(0)] = score

            pitch = row[2]
            pitch_padded[i, : pitch.size(0)] = pitch

            slurs = row[3]
            slurs_padded[i, : slurs.size(0)] = slurs

        return (
            phone_padded,
            phone_lengths,
            score_padded,
            pitch_padded,
            slurs_padded,
        )


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.

    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """

    def __init__(
        self,
        dataset,
        batch_size,
        boundaries,
        num_replicas=None,
        rank=None,
        shuffle=True,
    ):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries

        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas

    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)

        for i in range(len(buckets) - 1, 0, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i + 1)

        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (
                total_batch_size - (len_bucket % total_batch_size)
            ) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))

        batches = []
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            if (len_bucket == 0):
                continue
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]

            # add extra samples to make it evenly divisible
            rem = num_samples_bucket - len_bucket
            ids_bucket = (
                ids_bucket
                + ids_bucket * (rem // len_bucket)
                + ids_bucket[: (rem % len_bucket)]
            )

            # subsample
            ids_bucket = ids_bucket[self.rank:: self.num_replicas]

            # batching
            for j in range(len(ids_bucket) // self.batch_size):
                batch = [
                    bucket[idx]
                    for idx in ids_bucket[
                        j * self.batch_size: (j + 1) * self.batch_size
                    ]
                ]
                batches.append(batch)

        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        self.batches = batches

        assert len(self.batches) * self.batch_size == self.num_samples
        return iter(self.batches)

    def _bisect(self, x, lo=0, hi=None):
        if hi is None:
            hi = len(self.boundaries) - 1

        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x and x <= self.boundaries[mid + 1]:
                return mid
            elif x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            else:
                return self._bisect(x, mid + 1, hi)
        else:
            return -1

    def __len__(self):
        return self.num_samples // self.batch_size
