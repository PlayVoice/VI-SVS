# Init
Use VITS and Opencpop to develop singing voice synthesis; Maybe it will be VISinger.

# 本项目基于
https://github.com/jaywalnut310/vits
https://github.com/MoonInTheRiver/DiffSinger
https://wenet.org.cn/opencpop/

# 采样率转换
python wave_16k.py
--wavs
--wav_dump_16k

# 数据预处理
cd VISinger/

export PYTHONPATH=.

python prepare/data_vits.py
输出
1,../VISinger_data/label_vits/XXX._label.npy|XXX_score.npy|XXX_pitch.npy|XXX_slurs.npy
2,filelists/vits_file.txt 内容格式：wave path|label path|score path|pitch path|slurs path;

python prepare/preprocess.py
训练集随机打乱
验证集按排序

# VITS训练
# 使用16K节约内存，方便模型修改
# "learning_rate": 1e-4,# 原来是2e-4

CUDA_VISIBLE_DEVICES=0 python train.py -c configs/singing_base.json -m singing_base 2>exit_error.log;cat exit_error.log

# 测试验证

1,训练集生成验证:F0根据音频提取
python vsinging_debug.py

2,推理验证:F0根据规则生成
python vsinging_infer.py

3,完整歌曲合成
pyton vsinging_song.py

4,后验编码器验证
python vsinging_vae_debug.py -s ../VISinger_data/wav_dump_16k/*.wav

5,本项目完成冻结,F0的问题可以额外训练F0预测器,或者使用UTAU绘制pit曲线

![LOSS值](/resource/vising_loss.png)
![MEL谱](/resource/vising_mel.png)

<audio id="audio" controls="" preload="none">
      <source id="wav" src="/resource/vising_sample.wav">
</audio>

# 样例音频

[vits_singing_样例.wav](/resource/vising_sample.wav)