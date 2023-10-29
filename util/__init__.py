import numpy as np
import librosa
import pyworld

from svs import label_to_ids, load_midi_map, uv_map


class SingInput(object):
    def __init__(self, samplerate=32000, hop_size=320):
        self.fs = samplerate
        self.hop = hop_size
        self.notemaper = load_midi_map()

    def phone_to_uv(self, phones):
        uv = []
        for phone in phones:
            uv.append(uv_map[phone.lower()])
        return uv

    def notes_to_id(self, notes):
        note_ids = []
        for note in notes:
            note_ids.append(self.notemaper[note])
        return note_ids

    def frame_duration(self, durations):
        ph_durs = [float(x) for x in durations]
        sentence_length = 0
        for ph_dur in ph_durs:
            sentence_length = sentence_length + ph_dur
        sentence_length = int(sentence_length * self.fs / self.hop + 0.5)

        sample_frame = []
        startTime = 0
        for i_ph in range(len(ph_durs)):
            start_frame = int(startTime * self.fs / self.hop + 0.5)
            end_frame = int((startTime + ph_durs[i_ph]) * self.fs / self.hop + 0.5)
            count_frame = end_frame - start_frame
            sample_frame.append(count_frame)
            startTime = startTime + ph_durs[i_ph]
        all_frame = np.sum(sample_frame)
        assert all_frame == sentence_length
        # match mel length
        sample_frame[-1] = sample_frame[-1] - 1
        return sample_frame

    def score_duration(self, durations):
        ph_durs = [float(x) for x in durations]
        sample_frame = []
        for i_ph in range(len(ph_durs)):
            count_frame = int(ph_durs[i_ph] * self.fs / self.hop + 0.5)
            if count_frame >= 256:
                print("count_frame", count_frame)
                count_frame = 255
            sample_frame.append(count_frame)
        return sample_frame

    def parseInput(self, singinfo: str):
        infos = singinfo.split("|")
        file = infos[0]
        # hanz = infos[1]
        phon = infos[2].split(" ")
        note = infos[3].split(" ")
        note_dur = infos[4].split(" ")
        phon_dur = infos[5].split(" ")
        phon_slr = infos[6].split(" ")

        labels_ids = label_to_ids(phon)
        labels_uvs = self.phone_to_uv(phon)
        labels_frames = self.frame_duration(phon_dur)
        scores_ids = self.notes_to_id(note)
        scores_dur = self.score_duration(note_dur)
        labels_slr = [int(x) for x in phon_slr]
        return (
            file,
            labels_ids,
            labels_frames,
            scores_ids,
            scores_dur,
            labels_slr,
            labels_uvs,
        )

    def parseSong(self, singinfo: str):
        infos = singinfo.split("|")
        item_indx = infos[0]
        item_time = infos[1]
        # hanz = infos[2]
        phon = infos[3].split(" ")
        note_ids = infos[4].split(" ")
        note_dur = infos[5].split(" ")
        phon_dur = infos[6].split(" ")
        phon_slr = infos[7].split(" ")

        labels_ids = label_to_ids(phon)
        labels_uvs = self.phone_to_uv(phon)
        labels_frames = self.frame_duration(phon_dur)
        scores_ids = [int(x) if x != "rest" else 0 for x in note_ids]
        scores_dur = self.score_duration(note_dur)
        labels_slr = [int(x) for x in phon_slr]
        return (
            item_indx,
            item_time,
            labels_ids,
            labels_frames,
            scores_ids,
            scores_dur,
            labels_slr,
            labels_uvs,
        )

    def expandInput(self, labels_ids, labels_frames):
        assert len(labels_ids) == len(labels_frames)
        frame_num = np.sum(labels_frames)
        frame_labels = np.zeros(frame_num, dtype=np.int)
        start = 0
        for index, num in enumerate(labels_frames):
            frame_labels[start : start + num] = labels_ids[index]
            start += num
        return frame_labels

    def scorePitch(self, scores_id):
        score_pitch = np.zeros(len(scores_id), dtype=np.float)
        for index, score_id in enumerate(scores_id):
            if score_id == 0:
                score_pitch[index] = 0
            else:
                pitch = librosa.midi_to_hz(score_id)
                score_pitch[index] = round(pitch, 1)
        return score_pitch

    def smoothPitch(self, pitch):
        # 使用卷积对数据平滑
        kernel = np.hanning(5)  # 随机生成一个卷积核（对称的）
        kernel /= kernel.sum()
        smooth_pitch = np.convolve(pitch, kernel, "same")
        return smooth_pitch


class FeatureInput(object):
    def __init__(self, samplerate=32000, hop_size=320):
        self.fs = samplerate
        self.hop = hop_size

        self.f0_bin = 256
        self.f0_max = 1100.0
        self.f0_min = 50.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)

    def compute_f0(self, file):
        x, sr = librosa.load(file, sr=self.fs)
        assert sr == self.fs
        f0, t = pyworld.dio(
            x.astype(np.double),
            fs=sr,
            f0_ceil=900,
            frame_period=1000 * self.hop / sr,
        )
        f0 = pyworld.stonemask(x.astype(np.double), f0, t, self.fs)
        for index, pitch in enumerate(f0):
            f0[index] = round(pitch, 1)
        return f0

    def coarse_f0(self, f0):
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * (
            self.f0_bin - 2
        ) / (self.f0_mel_max - self.f0_mel_min) + 1

        # use 0 or 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > self.f0_bin - 1] = self.f0_bin - 1
        f0_coarse = np.rint(f0_mel).astype(np.int)
        assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (
            f0_coarse.max(),
            f0_coarse.min(),
        )
        return f0_coarse
