<div align="center">
<h1> Variational Inference with adversarial learning for end-to-end Singing Voice Synthesis </h1>

Different from VISinger, It is just VITS without MAS and DurationPredictor. 

迭代升级中~~~

基于nsf-bigvgan声码器和diffusion基音预测器
</div>

# 训练步骤
- 1 设置路径
> export PYTHONPATH=.

- 2 数据处理
> python prepare/data_vits.py

      生成文件../VISinger_data/label_vits/XXX._label.npy|XXX_score.npy|XXX_pitch.npy|XXX_slurs.npy

      生成文件filelists/vits_file.txt; 内容格式：wave path|label path|score path|pitch path|slurs path;

- 3 训练索引
> python prepare/preprocess.py

- 4 启动训练
> python train.py -c configs/singing_base.json -m singing_base

# 推理验证

- 1 训练集生成验证:F0根据音频提取

> python vsinging_debug.py

- 2 推理验证:F0根据规则生成

> python vsinging_infer.py

- 3 完整歌曲合成（[使用release模型](https://github.com/PlayVoice/VI-SVS/releases/tag/0.0.1)）

> pyton vsinging_song.py

- 4 F0的问题可以额外训练F0预测器,或者使用UTAU绘制pit曲线

![LOSS值](/resource/vising_loss.png)
![MEL谱](/resource/vising_mel.png)

# 样例音频

[vits_singing_样例.wav](/resource/vising_sample.wav)

# 参考项目
https://github.com/jaywalnut310/vits

https://wenet.org.cn/opencpop/

