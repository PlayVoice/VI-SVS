import numpy as np

from data_format_vits import label_for_vits

indx = "2001000001"
vits_label = label_for_vits(label_path=f'./VISinger/midis/{indx}.txt', midi_path=f'./VISinger/midis/{indx}.midi', fs=16000, hop=256)
frame_label = vits_label["frame_label"]
frame_pitch = vits_label["frame_pitch"]
np.save("./VISinger/midis/singing_label.npy", frame_label, allow_pickle=False)
np.save("./VISinger/midis/singing_pitch.npy", frame_pitch, allow_pickle=False)
