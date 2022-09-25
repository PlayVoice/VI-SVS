import os
import logging
import numpy as np
import librosa
import pyworld

from prepare.phone_map import label_to_ids
from prepare.phone_uv import uv_map


def load_midi_map():
    notemap = {}
    notemap["rest"] = 0
    fo = open("./prepare/midi-note.scp", "r+")
    while True:
        try:
            message = fo.readline().strip()
        except Exception as e:
            print("nothing of except:", e)
            break
        if message == None:
            break
        if message == "":
            break
        infos = message.split()
        notemap[infos[1]] = int(infos[0])
    fo.close()
    return notemap


class SingInput(object):
    def __init__(self, samplerate=16000, hop_size=128):
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
    def __init__(self, path, samplerate=16000, hop_size=128):
        self.fs = samplerate
        self.hop = hop_size
        self.path = path

        self.f0_bin = 256
        self.f0_max = 1100.0
        self.f0_min = 50.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)

    def compute_f0(self, filename):
        x, sr = librosa.load(self.path + filename, self.fs)
        assert sr == self.fs
        f0, t = pyworld.dio(
            x.astype(np.double),
            fs=sr,
            f0_ceil=800,
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

    def diff_f0(self, scores_pit, featur_pit, labels_frames):
        length_pit = min(len(scores_pit), len(featur_pit))
        offset_pit = np.zeros(length_pit, dtype=np.int)
        for idx in range(length_pit):
            s_pit = scores_pit[idx]
            f_pit = featur_pit[idx]
            if s_pit == 0 or f_pit == 0:
                offset_pit[idx] = 0
            else:
                tmp = int(f_pit - s_pit)
                tmp = +128 if tmp > +128 else tmp
                tmp = -127 if tmp < -127 else tmp
                tmp = 256 + tmp if tmp < 0 else tmp
                offset_pit[idx] = tmp
        offset_pit[offset_pit > 255] = 255
        offset_pit[offset_pit < 0] = 0
        # start = 0
        # for num in labels_frames:
        #     print("---------------------------------------------")
        #     print(scores_pit[start:start+num])
        #     print(featur_pit[start:start+num])
        #     print(offset_pit[start:start+num])
        #     start += num
        return offset_pit


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)  # ERROR & INFO

    notemaper = load_midi_map()
    logging.info(notemaper)

    singInput = SingInput(16000, 256)
    featureInput = FeatureInput("../VISinger_data/wav_dump_16k/", 16000, 256)

    if not os.path.exists("../VISinger_data/label_vits"):
        os.mkdir("../VISinger_data/label_vits")

    fo = open("../VISinger_data/transcriptions.txt", "r+")
    vits_file = open("./filelists/vits_file.txt", "w", encoding="utf-8")
    i = 0
    all_txt = []  # 统计非重复的句子个数
    while True:
        try:
            message = fo.readline().strip()
        except Exception as e:
            print("nothing of except:", e)
            break
        if message == None:
            break
        if message == "":
            break
        i = i + 1
        # if i > 5:
        #     exit()
        infos = message.split("|")
        file = infos[0]
        hanz = infos[1]
        all_txt.append(hanz)
        phon = infos[2].split(" ")
        note = infos[3].split(" ")
        note_dur = infos[4].split(" ")
        phon_dur = infos[5].split(" ")
        phon_slur = infos[6].split(" ")

        logging.info("----------------------------")
        logging.info(file)
        logging.info(hanz)
        logging.info(phon)
        # logging.info(note_dur)
        # logging.info(phon_dur)
        # logging.info(phon_slur)

        (
            file,
            labels_ids,
            labels_frames,
            scores_ids,
            scores_dur,
            labels_slr,
            labels_uvs,
        ) = singInput.parseInput(message)
        labels_ids = singInput.expandInput(labels_ids, labels_frames)
        labels_uvs = singInput.expandInput(labels_uvs, labels_frames)
        labels_slr = singInput.expandInput(labels_slr, labels_frames)
        scores_ids = singInput.expandInput(scores_ids, labels_frames)
        scores_pit = singInput.scorePitch(scores_ids)
        featur_pit = featureInput.compute_f0(f"{file}_bits16.wav")
        featur_pit = featur_pit[: len(labels_ids)]
        featur_pit = featur_pit * labels_uvs
        coarse_pit = featureInput.coarse_f0(featur_pit)

        # offset_pit = featureInput.diff_f0(scores_pit, featur_pit, labels_frames)
        assert len(labels_ids) == len(coarse_pit)

        logging.info(labels_ids)
        logging.info(scores_ids)
        logging.info(coarse_pit)
        logging.info(labels_slr)

        np.save(
            f"../VISinger_data/label_vits/{file}_label.npy",
            labels_ids,
            allow_pickle=False,
        )
        np.save(
            f"../VISinger_data/label_vits/{file}_score.npy",
            scores_ids,
            allow_pickle=False,
        )
        np.save(
            f"../VISinger_data/label_vits/{file}_pitch.npy",
            coarse_pit,
            allow_pickle=False,
        )
        np.save(
            f"../VISinger_data/label_vits/{file}_slurs.npy",
            labels_slr,
            allow_pickle=False,
        )

        # wave path|label path|label frame|score path|score duration;上面是一个.（当前目录），下面是两个..（从子目录调用）
        path_wave = f"../VISinger_data/wav_dump_16k/{file}_bits16.wav"
        path_label = f"../VISinger_data/label_vits/{file}_label.npy"
        path_score = f"../VISinger_data/label_vits/{file}_score.npy"
        path_pitch = f"../VISinger_data/label_vits/{file}_pitch.npy"
        path_slurs = f"../VISinger_data/label_vits/{file}_slurs.npy"
        print(
            f"{path_wave}|{path_label}|{path_score}|{path_pitch}|{path_slurs}",
            file=vits_file,
        )

    fo.close()
    vits_file.close()
    print(len(set(all_txt)))  # 统计非重复的句子个数
