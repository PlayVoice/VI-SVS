import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm
from torch.utils.data import DataLoader
from grad_extend.data import TextMelSpeakerDataset, TextMelSpeakerBatchCollate


filelist_path = "files/valid.txt"

dataset = TextMelSpeakerDataset(filelist_path)
collate = TextMelSpeakerBatchCollate()
loader = DataLoader(dataset=dataset, 
                    batch_size=2,
                    collate_fn=collate, 
                    drop_last=True,
                    num_workers=1, 
                    shuffle=True)

for batch in tqdm(loader):
    lengths = batch['lengths'].cuda()
    vec = batch['vec'].cuda()
    pit = batch['pit'].cuda()
    spk = batch['spk'].cuda()
    mel = batch['mel'].cuda()

    print('len', lengths.shape)
    print('vec', vec.shape)
    print('pit', pit.shape)
    print('spk', spk.shape)
    print('mel', mel.shape)