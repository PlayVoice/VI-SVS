import os, sys
sys.path.append(os.getcwd())
import logging
logging.basicConfig(level=logging.INFO)  # ERROR & INFO
import argparse
import numpy as np

from omegaconf import OmegaConf
from util import SingInput, FeatureInput


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--file', type=str, required=True)
    args = parser.parse_args()

    hps = OmegaConf.load(args.config)

    assert os.path.exists(args.file)
    assert os.path.exists(os.path.join(args.data, "wavs"))
    os.makedirs(os.path.join(args.data, "labels"), exist_ok=True)

    singInput = SingInput(hps.data.sampling_rate, hps.data.hop_length)
    featureInput = FeatureInput(hps.data.sampling_rate, hps.data.hop_length)

    raw_file = open(args.file, "r+")
    vits_file = open(os.path.join(args.data, "labels.txt"),
                     "w", encoding="utf-8")
    label_path = os.path.join(args.data, "labels")
    i = 0
    all_txt = []  # 统计非重复的句子个数
    while True:
        try:
            message = raw_file.readline().strip()
        except Exception as e:
            print("nothing of except:", e)
            break
        if message == None:
            break
        if message == "":
            break
        # i = i + 1
        # if i > 5:
        #    break
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
        path_wave = os.path.join(args.data, "wavs", f"{file}.wav")
        path_label = os.path.join(label_path, f"{file}_label.npy")
        path_score = os.path.join(label_path, f"{file}_score.npy")
        path_pitch = os.path.join(label_path, f"{file}_pitch.npy")
        path_slurs = os.path.join(label_path, f"{file}_slurs.npy")

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

        featur_pit = featureInput.compute_f0(path_wave)
        featur_pit = featur_pit[: len(labels_ids)]
        featur_pit = featur_pit * labels_uvs

        assert len(labels_ids) == len(featur_pit)

        np.save(path_label, labels_ids, allow_pickle=False)
        np.save(path_score, scores_ids, allow_pickle=False)
        np.save(path_pitch, featur_pit, allow_pickle=False)
        np.save(path_slurs, labels_slr, allow_pickle=False)

        print(
            f"{path_wave}|{path_label}|{path_score}|{path_pitch}|{path_slurs}",
            file=vits_file,
        )

    raw_file.close()
    vits_file.close()
    print(len(set(all_txt)))  # 统计非重复的句子个数
