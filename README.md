# Init
Use VITS and Opencpop to develop singing voice synthesis; Maybe it will VISinger.

# 本项目基于
https://github.com/jaywalnut310/vits    
https://github.com/SJTMusicTeam/Muskits/    
https://wenet.org.cn/opencpop/ 歌声数据 

# 使用muskit数据预处理，获得初步数据
cd egs/opencpop/svs1/       
./local/data.sh         

VISinger_data       
--lable     
--midi_dump     
--wav_dump      
# 采样率转换
python wave_16k.py      
--wav_dump      
--wav_dump_16k      
# 使用muskit将数据处理成vits的格式      
1, 将lable进行拆分      
python muskit/data_label_single.py      

label_dump,midi_dump,wav_dump:一个文件一个标注    

注意：label和lable的混用（两个单词都是对的）     

VISinger_data       
--label_dump        
--midi_dump     
--wav_dump      
--wav_dump_16k      

2, 将label和midi处理为frame对应的发音单元和音符（基音）     
python muskit/data_format_vits.py       
VISinger_data       
--label_vits        
--label_dump        
--midi_dump     
--wav_dump      
--wav_dump_16k      

3, 生成VITS需要的files，并分割为train和dev，test不需要（可以手动设计）      
python muskit/data_format_vits.py

vits_file.txt 中的内容格式：wave path|label path|pitch path;

cp vits_file.txt VISinger/filelists/        
cd VISinger/

python preprocess.py 分割为train和dev
# VITS训练

cd VISinger     
CUDA_VISIBLE_DEVICES=0 python train.py -c configs/singing_base.json -m singing_base 2>exit_error.log;cat exit_error.log     
python vsinging_infer.py

# 使用16K节约内存，方便模型修改

# 编辑midi，然后测试
cd ../;python muskit/infer_midi.py;cd -;python vsinging_edit.py

