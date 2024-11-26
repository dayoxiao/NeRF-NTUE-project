
# NTUE NeRF Implementation

  

### 此專案目標為，將含多視角資訊的現實圖片，以NeRF模型進行三維重建實踐，實現場景的任意新視角合成及模型抽取。

### 實踐內容包含：

透過colmap生成含位姿訊息的圖片資料。

以rays_util.py 從圖像（像素）進行射線（起點O和方向d）採樣。

model.py 進行位置編碼，將輸入透過sin, cos進行高維空間轉換。創建NeRF模型MLP架構，得以輸出預測密度sigma和顏色RGB。

render.py 在射線上進行採樣並得到真實座標，stratified從射線上初次得到採樣點，hierarchical則為透過計算累積分布函數(cdf)以weights比較大的區域進行二次精細採樣。

其中網絡輸出結果後處理raw2outputs，以網路輸出結果進行權重weight計算，並且轉化每條射線的顏色和深度。

主程式nerf.ipynb含參數設定及訓練步驟，以psnr(mse)為指標。

  

檢視程式eval.ipynb可供檢視測試及生成連續的彩色圖(RGB)以及深度圖(depth)。

  

若訓練場景為360度環視，則可以用extract_mesh.ipynb進行模型抽取。

  

# Installation

  

## Hardware

  

* OS: Windows 10

* Tested with NVIDIA GPU RTX3050 with **CUDA>=11.8**

  

## Software

  

* Setup Environment using anaconda is recommended

* This repo is test on Python=3.10

* Install other Python libraries by:

  

```

git clone https://github.com/dayoxiao/NeRF-NTUE-project

cd NeRF-NTUE-project

pip install -r requirements.txt

```
# Data Preprocessing



# Training

To start traning NeRF network after processing your raw data, put the output folder under the main NeRF-NTUE-project folder.

Currently in this repo, There are two type of scene you can train:


* [LLFF Forward Facing](#LLFF-Forward-Facing) 

* [LLFF 360 Inward Facing](#LLFF-360-Inward-Facing) 


## LLFF-Forward Facing

Take dataset name `fern` as example.

If your dataset is forward facing, change hyperparameters inside `nerf_train.ipynb` to following value:

```
expname = "fern_example"	#Your custom experiment name
data_dir = "./fern"			#Dataset directory
spherify = False
use_ndc = True

# Optional changes
factor = 0					# Load down scaled image, default not.
chunksize = 1024			# Modify as needed to fit in GPU memory.
display_rate = 100			# Frequency of displaying psnr value by iteration
save_rate = 1000			# Frequency of saving model weight by iteration.

```

## LLFF-360 Inward Facing

Take dataset name `mic` as example.

If your dataset is forward facing, change hyperparameters inside `nerf_train.ipynb` to following value:

```
expname = "mic_example"		#Your custom experiment name
data_dir = "./mic"			#Dataset directory
spherify = True
use_ndc = False

# Optional changes
factor = 4					# Load down scaled image, default is 4.
chunksize = 1024			# Modify as needed to fit in GPU memory.
display_rate = 100			# Frequency of displaying psnr value by iteration
save_rate = 1000			# Frequency of saving model weight by iteration.

```

# Evaluation
After training NeRF network, you should find several weight ckpts save under `./log/expname`.

In `eval.ipynb`, we reload these weight to generate synthetic views of RGB graph and depth graph.

To do so, make sure the hyperparameters inside `eval.ipynb` is the same as your training process in `nerf_train.ipynb`.

Furthermore, if needed,  you could change the `render_factor`  for faster downscale sampling output.

![image](https://github.com/dayoxiao/NeRF-NTUE-project/blob/yo_dev/pics/parrotnplate.gif)

# Mesh Extraction
Finally, If your scene is train in  LLFF-360 Inward Facing, you could try out `extract_mesh.ipynb` for reconstruction of 3D mesh.

Again, make sure model hyperparameters inside `extract_mesh.ipynb` is exact same as your training process in `nerf_train.ipynb`.

Then, follow the comment instruction inside the file to find the exact tight bounds for your scene to get the reconstruct result. You could export your work if needed.

![image](https://github.com/dayoxiao/NeRF-NTUE-project/blob/yo_dev/pics/parrot_mesh.png)

### 6/21未開發部分
自己的現實資料載入網路測試  
對網路的query生成多視圖，影片或gif輸出等  
360度視角生成能力確認，mesh抽取  

### 7/17更新
make_dataset.ipynb 製作dataset
進行影片切割成圖片，並將圖片縮小8倍
透過colmap生成圖片位資，並轉成llff格式資料集

load_llff.py 將llff格式資料集載入

### 8/19 更新 by:dayoxiao
與舊檔差別更新如下:
1. 移除訓練過程中檢驗測試圖片的步驟。因應舊程式使用llff資料同時進行訓練和檢驗會造成GPU記憶體空間不足 Out of Memory。
2. 加入NDC(Normalized Device Coordinates)轉換，以限定near, far值在0, 1之間。優化llff forward-facing資料訓練。
3. 加入Learning Rate Decay優化訓練結果。
4. 修改訓練迴圈並加入儲存模型Checkpoint的功能，現在可以暫停後繼續訓練以及儲存最終結果供後續測試調整。
5. 修正載入資料程式碼上錯誤分割取址，導致圖片複用造成的訓練結果不佳問題。

未完成部分:
1. 結果測試及檢驗。

### 9/13 更新 by:Lanlu
成功使用製作的資料集對模型做測試。
發財樹資料集訓練10000回合，PSNR如下
![image](https://github.com/dayoxiao/NeRF-NTUE-project/blob/main/test_psnr.jpg)

### 11/16 更新 by:dayoxiao
增加檔案eval，用以渲染彩色圖等最終成果。

### 11/25 更新 by:dayoxiao
新增檔案extract_mesh 現在可以進行3D模型抽取了。
eval檔除了rgb圖現在還會產生depth圖。

---------------------------------------

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
--Calculate weights based on the network output and transform to the color map and depth map.  

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

Original image:
![image](https://github.com/dayoxiao/NeRF-NTUE-project/blob/yo_dev/pics/original_img.png)

Step Training:  
![image](https://github.com/dayoxiao/NeRF-NTUE-project/blob/yo_dev/pics/showcase.gif)

Result:  
![image](https://github.com/dayoxiao/NeRF-NTUE-project/blob/yo_dev/pics/final%20result.png)
