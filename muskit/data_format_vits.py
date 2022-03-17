import os
import sys
import numpy as np
import miditoolkit
import logging

from midi_load import read_label_duration, midi_to_seq
from phone_map import label_to_ids


def label_for_vits(label_path:str, midi_path:str, fs=24000, hop=256):
    # the format of phoneme duration information
    label_duration_file = open(label_path, "r", encoding="utf-8")
    label_duration_info = label_duration_file.read()
    logging.debug("duration: {}".format(label_duration_info))
    # get phone label and its duration
    labels, durations = read_label_duration(label_duration_info)
    labels_ids = label_to_ids(labels)
    logging.info(labels)
    logging.info(labels_ids)
    logging.info(durations)

    # load music score and conduct preprocessing
    midi_obj = miditoolkit.midi.parser.MidiFile(midi_path)
    logging.debug(midi_obj)
    for note in midi_obj.instruments[0].notes:
        logging.info(note)
    notes, tempos = midi_to_seq(midi_obj, np.int16, fs)
    logging.info(np.size(notes)) # samples 第一个note开始和最后一个note结束覆盖的采样点数，小于总音频长度
    logging.info(np.size(tempos)) # samples 第一个note开始和最后一个note结束覆盖的采样点数，小于总音频长度

    # input information
    # info = {
    #     "label": labels_ids,
    #     "duration": durations,
    #     "note": notes, #  data["score"]
    #     "tempo": tempos # data["tempo"]
    # }
    # logging.info(info)

    logging.info("=====================================================================================")
    logging.info("采样点数据信息")

    len_notes = len(notes)
    len_waves = int(durations[-1][1] * fs)
    len_max = max(len_notes, len_waves)

    pitchseq = np.zeros(len_max)
    pitchseq[:len_notes] = notes

    # [Shuai]: length of label - phone_id seq is the same of midi,
    labelseq = np.zeros(len_max)
    for i in range(durations.shape[0]):# time repair
        start = int(durations[i, 0] * fs)
        end = int(durations[i, 1] * fs) + 1
        if end > len_max:
            end = len_max - 1
        labelseq[start:end] = labels_ids[i]

    pitchseq.astype(np.int64)
    labelseq.astype(np.int64)

    # input sample level information
    data_sample = {
        "sample_label": labelseq,
        "sample_pitch": pitchseq,
    }
    logging.info(np.size(labelseq))
    logging.info(np.size(pitchseq))
    logging.info(data_sample)

    logging.info("=====================================================================================")
    logging.info("语音帧数据信息")

    len_hop = hop
    len_frame = len_max // len_hop
    label_frames = np.zeros(len_frame)
    pitch_frames = np.zeros(len_frame)

    for i in range(len_frame):
        label_frames[i] = labelseq[i * len_hop]
        pitch_frames[i] = pitchseq[i * len_hop]

    label_frames.astype(np.int64)
    pitch_frames.astype(np.int64)

    # input frame level information
    data_frame = {
        "frame_label": label_frames,
        "frame_pitch": pitch_frames,
    }

    logging.info(np.size(label_frames))
    logging.info(np.size(pitch_frames))
    logging.info(data_frame)
    return data_frame

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    test = label_for_vits(label_path='muskit/2001000001.txt', midi_path='muskit/2001000001.midi', fs=16000, hop=256)
    np.save("test_vits_label.npy", test["frame_label"], allow_pickle=False)
    np.save("test_vits_pitch.npy", test["frame_pitch"], allow_pickle=False)

    test_load_label = np.load("test_vits_label.npy")
    test_load_pitch = np.load("test_vits_pitch.npy")
    print(test_load_label)
    print(test_load_pitch)

    if not os.path.exists("./VISinger_data/label_vits"):
        os.mkdir("./VISinger_data/label_vits")

    vits_file = open('vits_file.txt', 'w', encoding='utf-8')
    for label_file in os.listdir("./VISinger_data/label_dump"):
        if (label_file.endswith(".lab")):
            fileid = label_file[:-4]
            print(fileid)
            data_frame = label_for_vits(label_path=f'./VISinger_data/label_dump/{fileid}.lab',
                                        midi_path=f'./VISinger_data/midi_dump/{fileid}.midi', fs=16000, hop=256)
            np.save(f"./VISinger_data/label_vits/{fileid}_label.npy", data_frame["frame_label"], allow_pickle=False)
            np.save(f"./VISinger_data/label_vits/{fileid}_pitch.npy", data_frame["frame_pitch"], allow_pickle=False)
            # wave path|label path|pitch path;上面是一个.（当前目录），下面是两个..（从子目录调用）
            path_wave = f"../VISinger_data/wav_dump_16k/{fileid}_bits16.wav"
            path_label = f"../VISinger_data/label_vits/{fileid}_label.npy"
            path_pitch = f"../VISinger_data/label_vits/{fileid}_pitch.npy"
            print(f"{path_wave}|{path_label}|{path_pitch}", file=vits_file)
    vits_file.close()