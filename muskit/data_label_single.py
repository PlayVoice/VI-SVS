import os


if not os.path.exists("./VISinger_data/label_dump"):
    os.mkdir("./VISinger_data/label_dump")

for path in ["train", "dev", "test"]:
    fo = open(f'./VISinger_data/lable/{path}/label', 'r', encoding='utf-8')
    while(True):
        try:
            label_item = fo.readline().strip()
        except Exception as e:
            print('nothing of except:', e)
            break
        if (label_item == None):
            break
        if (label_item == ""):
            break
        label_split = label_item.split(' ', 1)
        with open(f"./VISinger_data/label_dump/{label_split[0]}.lab", "w", encoding='utf-8') as fout:
            print(f"{label_split[1]}", file=fout)
    fo.close()

