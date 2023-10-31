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
```
python util/resample.py -w segments/wavs/ -o data_svs/wavs -s 32000
```

- 3 生成数据标注
```
python util/generate_label.py --config configs/singing_base.yaml --data data_svs/ --file segments/transcriptions.txt
```

data_svs/labels.txt，内容格式：wave path|label path|score path|pitch path|slurs path

- 3 划分训练索引
```
python util/generate_label.py --file data_svs/labels.txt
```

生成 filelists/singing_train.txt 和 filelists/singing_valid.txt

- 4 启动训练
```
python svs_train.py -c configs/singing_base.yaml -n vits_svs
```

# 推理验证

- 0 模型导出
```
python svs_export.py --config configs/singing_base.yaml --model chkpt/vits_svs/vits_svs_****.pt
```

- 1 推理验证: F0根据乐谱生成，**基于Diffusion的音高预测还未完成**
```
python svs_infer.py --config configs/singing_base.yaml --model svs_opencpop.pt
```

- 2 完整歌曲合成（[使用release模型](https://github.com/PlayVoice/VI-SVS/releases/tag/0.0.2)）
```
python svs_song.py --config configs/singing_base.yaml --model svs_opencpop.pt
```

- 3 TODO **Diffusion Pitch**

# 参考项目
https://github.com/jaywalnut310/vits

https://wenet.org.cn/opencpop/

https://github.com/NVIDIA/BigVGAN

https://github.com/mindslab-ai/univnet

https://github.com/PlayVoice/so-vits-svc-5.0

https://github.com/shivammehta25/Matcha-TTS

https://github.com/MoonInTheRiver/DiffSinger

[RoFormer: Enhanced Transformer with rotary position embedding](https://arxiv.org/abs/2104.09864)

