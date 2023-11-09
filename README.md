<div align="center">
<h1> Variational Inference with adversarial learning for end-to-end Singing Voice Synthesis </h1>

Different from VISinger, It is just VITS without MAS and DurationPredictor. 

作为一个用于学习的项目，就这样了：Pitch的预测是需要改进的地方

![VISinger](https://github.com/MaxMax2016/VI-SVS/assets/16432329/c76ca716-b230-4852-b8f0-2c3041af7072)

![VI-SVS](https://github.com/MaxMax2016/VI-SVS/assets/16432329/128c0f33-4428-4b57-9cd3-b6237f53c1a4)

</div>

**Pitch and Duration will be developed as add-on!**

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

- 5 训练Pitch
```
python pit_train.py -c configs/singing_base.yaml -n pitch
```

# 推理验证

- 0 模型导出
```
python svs_export.py --config configs/singing_base.yaml --model chkpt/vits_svs/vits_svs_****.pt
```

- 1 推理验证: F0根据乐谱生成
```
python svs_infer.py --config configs/singing_base.yaml --model svs_opencpop.pt
```

- 2 完整歌曲合成（[使用release模型](https://github.com/PlayVoice/VI-SVS/releases/tag/0.0.2)）
```
python svs_song.py --config configs/singing_base.yaml --model svs_opencpop.pt
```

# 推理验证，使用Pitch预测，效果不佳

- 0 模型导出
```
python svs_export.py --config configs/singing_base.yaml --model chkpt/vits_svs/vits_svs_****.pt
```

```
python pit_export.py --config configs/singing_base.yaml --model chkpt/pitch/pitch_****.pt
```

- 1 推理验证
```
python svs_infer_pitch.py --config configs/singing_base.yaml --model svs_opencpop.pt --pitch pit_opencpop.pt
```

- 2 完整歌曲合成（[使用release模型](https://github.com/PlayVoice/VI-SVS/releases/tag/0.0.2)）
```
python svs_song_pitch.py --config configs/singing_base.yaml --model svs_opencpop.pt --pitch pit_opencpop.pt
```

# 数据

https://wenet.org.cn/opencpop/

# 歌声合成参考

https://github.com/SJTMusicTeam/Muskits

https://github.com/MoonInTheRiver/DiffSinger

[VISinger: Variational Inference with Adversarial Learning for End-to-End Singing Voice Synthesis](https://arxiv.org/abs/2110.08813)

# 模型设计参考

https://github.com/NVIDIA/BigVGAN

https://github.com/jaywalnut310/vits

https://github.com/mindslab-ai/univnet

https://github.com/PlayVoice/so-vits-svc-5.0

https://github.com/shivammehta25/Matcha-TTS

[RoFormer: Enhanced Transformer with rotary position embedding](https://arxiv.org/abs/2104.09864)

# Diffusion Pitch

https://github.com/thuhcsi/DiffVar

https://github.com/hayeong0/Diff-HierVC

https://github.com/tonnetonne814/SiFi-VITS2-44100-Ja

[Grad-TTS: A Diffusion Probabilistic Model for Text-to-Speech](https://arxiv.org/abs/2105.06337)

# Diffusion Pitch of Diff-HierVC
![DiffPitch](https://github.com/PlayVoice/VI-SVS/assets/16432329/055d75a4-7009-46c1-8603-65254cec47dd)
