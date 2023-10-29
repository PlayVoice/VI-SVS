from svs.phone_map import label_to_ids
from svs.phone_uv import uv_map


def load_midi_map():
    notemap = {}
    notemap["rest"] = 0
    fo = open("./svs/midi-note.scp", "r+")
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
