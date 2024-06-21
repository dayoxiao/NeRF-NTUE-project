# NTUE NeRF Implementation

### 此專案目標為，將含多視角資訊的現實圖片，以NeRF模型進行三維重建實踐，實現場景的任意新視角合成。  
### 實踐內容包含：  
透過colmap生成含位姿訊息的圖片資料。  
  
以rays_util.py 從圖像（像素）進行射線（起點O和方向d）採樣。
  
model.py 進行位置編碼，將輸入透過sin, cos進行高維空間轉換。創建NeRF模型MLP架構，得以輸出預測密度sigma和顏色RGB。
  
render.py 在射線上進行採樣並得到真實座標，stratified從射線上初次得到採樣點，hierarchical則為透過計算累積分布函數（cdf)以weights比較大的區域進行二次精細採樣。
  
其中網絡輸出結果後處理raw2outputs，以網路輸出結果進行權重weight計算，並且轉化每條射線的顏色圖和深度圖。
  
主程式nerf.ipynb含參數設定及訓練步驟，以psnr(mse)為指標。

### 6/21未開發部分
自己的現實資料載入網路測試
對網路的query生成多視圖，影片或gif輸出等
360度視角生成能力確認，mesh抽取

### Project Objective
The goal of this project is to achieve 3D reconstruction of real-world images containing multi-view information using the NeRF model, enabling arbitrary new view synthesis of the scene.

### Implementation Details Include:
Generating Images with Pose Information using colmap:  
--Utilize colmap to generate images that include pose information.  
  
Sampling Rays from Images (Pixels) using rays_util.py:  
--Extract rays (with starting point O and direction d) from the images.  
  
Position Encoding in model.py:  
--Perform position encoding by transforming inputs into high-dimensional space using sine and cosine functions.  
--Create the NeRF model MLP architecture to output predicted density sigma and color RGB.  

Sampling and Real Coordinate Acquisition in render.py:  
--Perform sampling along the rays to obtain real coordinates.  
--The stratified method initially samples points along the ray, while the hierarchical method performs secondary fine sampling in regions with larger weights by calculating the cumulative distribution function (CDF). 
Post-processing of Network Output in raw2outputs:  
--Calculate weights based on the network output and transform the color map and depth map of each ray.  

Main Program nerf.ipynb:  
--Contains parameter settings and training steps, with psnr (mse) as the evaluation metric.  
  
### Undeveloped Parts before 6/21
Loading and testing real-world data on the network.  
Generating multi-view queries for the network, such as outputting videos or gifs.  
Confirming the ability to generate 360-degree views and extracting mesh.  

## Architecture

```
nerf.ipynb.........main python training file  
  --rays_util.py...sample rays
    --get_rays
  --model.py.......NN structure
    --PositionalEncoder
    --NeRF
  --render.py......sample points & ouput
    --sample_stratified
    --raw2outputs
    --sample_pdf
    --sample_hierarchical
```

## Showcase

Step Training:  
![image](https://github.com/dayoxiao/NeRF-NTUE-project/blob/yo_dev/pics/showcase.gif)

Result:  
![image](https://github.com/dayoxiao/NeRF-NTUE-project/blob/yo_dev/pics/final%20result.png)
