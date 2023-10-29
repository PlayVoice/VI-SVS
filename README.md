<div align="center">
<h1> Variational Inference with adversarial learning for end-to-end Singing Voice Synthesis </h1>

Different from VISinger, It is just VITS without MAS and DurationPredictor. 

迭代升级中~~~

</div>

# 训练步骤

- 1 下载数据 segments.zip，并解压

```
segments
|-- test.txt
|-- train.txt
|-- transcriptions.txt
`-- wavs
    |-- 2001000001.wav
    |-- 2001000002.wav
    |-- 2001000003.wav
```

- 2 转换采样率: 本项目采用32KHz

> python util/resample.py -w segments/wavs/ -o data_svs/wavs -s 32000 

- 3 生成数据标注

> python util/generate_label.py --config configs/singing_base.json --data data_svs/ --file segments/transcriptions.txt

data_svs/labels.txt，内容格式：wave path|label path|score path|pitch path|slurs path

- 3 划分训练索引

> python util/generate_label.py --file data_svs/labels.txt

生成 filelists/singing_train.txt 和 filelists/singing_valid.txt

- 4 启动训练

> python svs_train.py -c configs/singing_base.yaml -n vits_svs

# 推理验证

- 0 模型导出

> python svs_export.py --config configs/singing_base.yaml --model chkpt/vits_svs/vits_svs_****.pt

- 1 推理验证:F0根据规则生成

> python svs_infer.py --config configs/singing_base.yaml --model svs_opencpop.pt 

- 2 完整歌曲合成（[使用release模型](https://github.com/PlayVoice/VI-SVS/releases/tag/0.0.1)）

> python svs_song.py --config configs/singing_base.yaml --model svs_opencpop.pt

- 3 TODU Diffusion Pitch

![LOSS值](/resource/vising_loss.png)
![MEL谱](/resource/vising_mel.png)

# 参考项目
https://github.com/jaywalnut310/vits

https://wenet.org.cn/opencpop/

