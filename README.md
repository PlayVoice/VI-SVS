# Init
Use VITS and Opencpop to develop singing voice synthesis; 
Different from VISinger, It is just VITS without MAS and DurationPredictor.

# 本项目基于
https://github.com/jaywalnut310/vits

https://github.com/MoonInTheRiver/DiffSinger

https://wenet.org.cn/opencpop/

# 数据预处理
export PYTHONPATH=.

python prepare/data_vits.py

      生成文件../VISinger_data/label_vits/XXX._label.npy|XXX_score.npy|XXX_pitch.npy|XXX_slurs.npy

      生成文件filelists/vits_file.txt; 内容格式：wave path|label path|score path|pitch path|slurs path;

python prepare/preprocess.py

# VITS训练

python train.py -c configs/singing_base.json -m singing_base

# 测试验证

1,训练集生成验证:F0根据音频提取

python vsinging_debug.py

2,推理验证:F0根据规则生成

python vsinging_infer.py

3,完整歌曲合成（**使用release模型**）

pyton vsinging_song.py

4,F0的问题可以额外训练F0预测器,或者使用UTAU绘制pit曲线


![LOSS值](/resource/vising_loss.png)
![MEL谱](/resource/vising_mel.png)

<audio id="audio" controls="" preload="none">
      <source id="wav" src="/resource/vising_sample.wav">
</audio>

# 样例音频

[vits_singing_样例.wav](/resource/vising_sample.wav)

# AI修复
https://github.com/brentspell/hifi-gan-bwe
