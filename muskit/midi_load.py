import miditoolkit
import numpy as np


def read_label_duration(label_str):
    line = label_str.strip().split()
    seq_len = len(line) // 3
    sample_time = np.zeros((seq_len, 2))
    sample_label = []
    for i in range(seq_len):
        sample_label.append(line[i * 3 + 2])
        sample_time[i, 0] = np.float32(line[i * 3])
        sample_time[i, 1] = np.float32(line[i * 3 + 1])
    return sample_label, sample_time


def get_tick_to_time_mapping(ticks_per_beat, tempo_changes, max_tick=np.int32(1e6)):
    """
    Get mapping from ticks to seconds with tempo information
    """
    tick_to_time = np.zeros(max_tick + 1)
    num_tempi = len(tempo_changes)

    fianl_tick = max_tick
    acc_time = 0

    for idx in range(num_tempi):
        start_tick = tempo_changes[idx].time
        cur_tempo = int(tempo_changes[idx].tempo)

        # compute tick scale
        seconds_per_beat = 60 / cur_tempo
        seconds_per_tick = seconds_per_beat / float(ticks_per_beat)

        # set end tick of interval
        end_tick = tempo_changes[idx + 1].time if (idx + 1) < num_tempi else fianl_tick

        # wrtie interval
        ticks = np.arange(end_tick - start_tick + 1)
        tick_to_time[start_tick : end_tick + 1] = acc_time + seconds_per_tick * ticks
        acc_time = tick_to_time[end_tick]
    return tick_to_time


def midi_to_seq(
    midi_obj, dtype=np.int16, rate=24000, time_aug_factor=1
):
    """method for midi_obj.
    Input:
        miditoolkit_object, sampling rate
    Output:
        note_seq: np.array([pitch1,pitch2....]), which length is equal to note.time*rate
        tempo_seq:np.array([pitch1,pitch2....]), which length is equal to note.time*rate
    """
    tick_to_time = midi_obj.get_tick_to_time_mapping()
    max_time = tick_to_time[-1]
    # print(max_time)
    notes = midi_obj.instruments[0].notes
    notes.sort(key=lambda x: (x.start, x.pitch))
    # print(notes)

    tempos = midi_obj.tempo_changes
    tempos.sort(key=lambda x: (x.time, x.tempo))
    # print(tempos)

    assert len(tempos) == 1     # NOTE(Shuai): the len(tempos) will be 1 if dataset are prepared successfully (kiritan, natsume, oniku, ofuton)

    # used for Format data or Training other acoustic model like RNN
    note_seq = np.zeros(int(rate * max_time * time_aug_factor), dtype=dtype)
    for i in range(len(notes)):
        st = int(tick_to_time[notes[i].start] * rate * time_aug_factor)
        ed = int(tick_to_time[notes[i].end] * rate * time_aug_factor)
        note_seq[st:ed] = notes[i].pitch

    tempo_seq = np.zeros(int(rate * max_time * time_aug_factor), dtype=dtype)
    for i in range(len(tempos) - 1):
        st = int(tick_to_time[tempos[i].time] * rate * time_aug_factor)
        ed = int(tick_to_time[tempos[i + 1].time] * rate * time_aug_factor)
        tempo_seq[st:ed] = int(tempos[i].tempo + 0.5)
    st = int(tick_to_time[tempos[-1].time] * rate * time_aug_factor)
    tempo_seq[st:] = int(tempos[-1].tempo + 0.5)

    return note_seq, tempo_seq


def seq_to_midi(
    note_seq,
    tempo_seq,
    rate=24000,
    DEFAULT_RESOLUTION=960,
    DEFAULT_TEMPO=120,
    DEFAULT_VELOCITY=64,
):
    """method for note_seq.
    Input:
        note_seq, tempo_seq, sampling rate
    Output:
        miditoolkit_object with default resolution, tempo and velocity.
    """
    # get downbeat and note (no time)
    temp_notes = note_seq
    temp_tempos = tempo_seq
    ticks_per_beat = DEFAULT_RESOLUTION

    # get specific time for tempos
    tempos = []
    i = 0
    last_i = 0
    # acc_time = 0
    acc_tick = 0
    while i < len(temp_tempos):
        bpm = temp_tempos[i]
        ticks_per_second = DEFAULT_RESOLUTION * bpm / 60
        j = i
        while j + 1 < len(temp_tempos) and temp_tempos[j + 1] == bpm:
            j += 1
        if bpm == 0:
            bpm = DEFAULT_TEMPO
        tempos.append(miditoolkit.midi.containers.TempoChange(bpm, acc_tick))
        acc_tick += int((j - last_i + 1) * ticks_per_second / rate)

        last_i = j
        i = j + 1
    tick_to_time = get_tick_to_time_mapping(ticks_per_beat, tempos)

    # get specific time for notes
    notes = []
    i = 0
    while i < len(temp_notes):
        pitch = temp_notes[i]
        j = i
        while j + 1 < len(temp_notes) and temp_notes[j + 1] == pitch:
            j += 1
        st = i / rate
        ed = j / rate

        start = np.searchsorted(tick_to_time, st, "left")
        end = np.searchsorted(tick_to_time, ed, "left")
        if pitch > 0 and pitch <= 128:
            notes.append(
                miditoolkit.midi.containers.Note(
                    start=start, end=end, pitch=pitch, velocity=DEFAULT_VELOCITY
                )
            )

        i = j + 1

    # write
    midi = miditoolkit.midi.parser.MidiFile()
    midi.ticks_per_beat = DEFAULT_RESOLUTION
    # write instrument
    inst = miditoolkit.midi.containers.Instrument(0, is_drum=False)
    inst.notes = notes
    midi.instruments.append(inst)
    # write tempo
    midi.tempo_changes = tempos
    return midi


if __name__ == "__main__":
    import os

    import miditoolkit

    path = "test.midi"
    midi_obj = miditoolkit.midi.parser.MidiFile(path)

    note_seq, tempo_seq = midi_to_seq(midi_obj, np.int16, np.int16(24000))

    print(f"note_seq: {note_seq[10000]}, note_seq.shape: {note_seq.shape}")

    note_list = []
    rootpath = "/data5/gs/Muskits/egs/kiritan/svs1/dump/raw/org/tr_no_dev/data"

    for index in range(1, 33):
        folderName = f"format_midi.{str(index)}"
        folderPath = os.path.join(rootpath, folderName)
        for filename in os.listdir(folderPath):
            if filename.endswith(".midi"):
                midiPath = os.path.join(folderPath, filename)

                midi_obj = miditoolkit.midi.parser.MidiFile(midiPath)
                note_seq, tempo_seq = midi_to_seq(midi_obj, np.int16, np.int16(24000))

                note_list.append(note_seq)
                # print(np.mean(note_seq))
    res = np.hstack(note_list)
    print(f"shape: {res.shape}, mean: {np.mean(res)}")
