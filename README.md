基於連續頻率控制的音頻合成與和諧度評估系統
===  

  

這個項目旨在通過連續控制音頻頻率，而不是依賴於傳統的十二平均律音符，來生成和諧的音調（和聲）和旋律。該系統包括音頻合成、和諧度評分以及基於評分的頻率序列生成方法，從而實現在實際音樂創作中的應用。


        
### 功能介紹
#### 1. 音頻合成 (`sound_object.py` 與 `supporting_functions.py`) 
  `Sound` 物件利用多種參數控制聲音的合成，包括頻率、音量、音色（`TimbreFactor`），並支持透過 ADSR（Attack, Decay, Sustain, Release）模型對音頻的包络進行調整。
  支持調製波形，通過頻率調製（Frequency Modulation，FM）添加複雜度和豐富度到聲音中。
  包含保存合成音頻為WAV文件的功能。
#### 2. 和諧度評分與頻率調整 (`ratio_scoring.py`)
  提供和諧度評分機制，根據頻率比例的簡單性和音樂上的和諧度給出分數。
  包含查找特定和諧度範圍內的頻率比例以及基於目標和諧度或頻率變動目標來調整頻率比的方法。
#### 3. 基於頻率控制的作曲方法 (`composing_by_frequency.py`)
  提供了一種基於頻率序列生成的方法，用於創作旋律和和聲，允許更精細的控制音樂的和諧度和表達性。

### 使用指南

#### 安裝

確保您的系統已安裝Python 3.x以及以下模組：

    numpy  
    scipy  
    matplotlib
    
## 快速入門

#### 初始化一個基本的聲音物件
    first_sound = Sound(length=1, freq=442)
#### 將一個以上的聲音物件合併輸出
    a = Sound(length=10, freq=[442, 438, 430], vol=[30, 30, 20])
    b = Sound(length=5, starts_at=2.5, freq=[221, 201, 215], vol=[20, 15, 20])
    output_sound_objs([a, b])
#### 使用ratio_scoring.py中的功能來評估和諧度或進行頻率調整


