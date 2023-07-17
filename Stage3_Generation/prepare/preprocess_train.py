import os
import random


if __name__ == "__main__":
    os.makedirs("./files/", exist_ok=True)

    rootPath = "./data_bigvgan/waves-32k/"
    all_items = []
    for spks in os.listdir(f"./{rootPath}"):
        if os.path.isdir(f"./{rootPath}/{spks}"):
            for file in os.listdir(f"./{rootPath}/{spks}"):
                if file.endswith(".wav"):
                    file = file[:-4]
                    path_wave = f"./data_bigvgan/waves-32k/{spks}/{file}.wav"
                    path_pitch = f"./data_bigvgan/pitch/{spks}/{file}.pit.npy"
                    path_mel = f"./data_bigvgan/mel/{spks}/{file}.mel.pt"

                    assert os.path.isfile(path_wave), path_wave
                    assert os.path.isfile(path_pitch), path_pitch
                    assert os.path.isfile(path_mel), path_mel
                    all_items.append(f"{path_wave}|{path_pitch}|{path_mel}")

    random.shuffle(all_items)
    valids = all_items[:10]
    valids.sort()
    trains = all_items[10:]
    # trains.sort()
    fw = open("./files/valid.txt", "w", encoding="utf-8")
    for strs in valids:
        print(strs, file=fw)
    fw.close()
    fw = open("./files/train.txt", "w", encoding="utf-8")
    for strs in trains:
        print(strs, file=fw)
    fw.close()
