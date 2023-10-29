import os
import random
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True)
    args = parser.parse_args()
    alls = []
    fo = open(args.file, "r+")
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
        alls.append(message)
    fo.close()

    valids = alls[:5]
    trains = alls[5:]

    random.shuffle(trains)
    os.makedirs("filelists", exist_ok=True)

    fw = open("./filelists/singing_valid.txt", "w", encoding="utf-8")
    for strs in valids:
        print(strs, file=fw)
    fw.close()

    fw = open("./filelists/singing_train.txt", "w", encoding="utf-8")
    for strs in trains:
        print(strs, file=fw)
    fw.close()
