# "Hand Gesture Recognition"
## 概要
OpenCVを用いて作成したジェスチャー認識のプログラムです。ビデオカメラに映った手が示すジェスチャーをリアルタイムで認識し、ジェスチャーに対応する名前と画像を映像上に表示します。顔認識も行っているため、顔が同時に写っていても影響は小さいです。

This is a gesture recognition program created using OpenCV. It recognizes the hand gestures on a video camera. The name and image are displayed on the video in real time. Face recognition is also performed, so even if the faces are in the picture at the same time, the effect is minimal.  

__動作の様子：demonstrate.mp4__

## 起動
まずOpenCVとC++が必要です。
```
$ git clone https://github.com/markchen0/hand-gesture.git
$ cd hand-gesture
$ cd src
$ make gesture
$ ./gesture
```
## 操作　
まず、自分が画面に写っていない状態でスペースキーを押してください。その後、カメラの正面で手の形を作ってみてください。

また、顔がうまく認識されない場合は、顔が写り込まないようにすると精度が上がります。

照明や照明の反射、肌色に近い物体が背景に映り込んでいる場合は精度が落ちてしまいます。

グー・人差し指・グッド・ピース・きつねの形・３・OK・４・パー・つかむ（パーからすべての指を少し曲げた状態）を認識します。
 
スペース： 背景差分に使う背景画像の更新  
esc: 終了


First, press the spacekey while you are not in the screen. Then try to make a hand shape in front of the camera.

Also, if your face is not well recognized, you can improve your accuracy by keeping your face out of the picture.

Lighting, lighting reflections, and objects close to your skin tone in the background will reduce the accuracy.

Recognize 10 gestures below.
・rock, 1, good, piece sign, fox, 3, OK, 4, open hand, grab (all fingers slightly bent)
 
Space: Update the background image used for background differences.  
esc: quit
