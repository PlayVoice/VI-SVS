import os
import random


def print_error(info):
    print(f"\033[31m File isn't existed: {info}\033[0m")


if __name__ == "__main__":
    os.makedirs("./files/", exist_ok=True)

    rootPath = "./data_gvc/waves-32k/"
    all_items = []
    for spks in os.listdir(f"./{rootPath}"):
        if not os.path.isdir(f"./{rootPath}/{spks}"):
            continue
        print(f"./{rootPath}/{spks}")
        for file in os.listdir(f"./{rootPath}/{spks}"):
            if file.endswith(".wav"):
                file = file[:-4]

                path_mel = f"./data_gvc/mel/{spks}/{file}.mel.pt"
                path_vec = f"./data_gvc/hubert/{spks}/{file}.vec.npy"
                path_pit = f"./data_gvc/pitch/{spks}/{file}.pit.npy"
                path_spk = f"./data_gvc/speaker/{spks}/{file}.spk.npy"

                has_error = 0
                if not os.path.isfile(path_mel):
                    print_error(path_mel)
                    has_error = 1
                if not os.path.isfile(path_vec):
                    print_error(path_vec)
                    has_error = 1
                if not os.path.isfile(path_pit):
                    print_error(path_pit)
                    has_error = 1
                if not os.path.isfile(path_spk):
                    print_error(path_spk)
                    has_error = 1
                if has_error == 0:
                    all_items.append(
                        f"{path_mel}|{path_vec}|{path_pit}|{path_spk}")

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
