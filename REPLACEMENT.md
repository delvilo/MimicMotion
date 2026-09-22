# 保留原背景與運鏡的人物替換

本版預設為 `--mode replace`。原先整幅生成模式保留為 `--mode generate`。

本版是可檢查的處理流程，不是「完美換人」保證。尚未在完整模型環境完成真實影片驗收。MimicMotion 仍負責合成人物，因此可能有身份、手腳和動作細節偏差；SAM 2 可能有分割邊緣或追蹤誤差。

## 實際流程

1. 逐幀讀取原舞蹈片，保留原始幀数、解析度、鏡頭座標和幀率；不再抽幀，也不把舞蹈座標縮放到參考人物的身材位置。
2. DWPose 提取指定主角動作；MimicMotion 生成替換人物。生成畫面的背景不作為合成底圖。
3. SAM 2 分別追蹤原主角與生成主角，取得人物遮罩。可用人工精修遮罩取代自动結果。
4. 合成底圖先使用當前原片幀。只對「原主角原本佔據、但新主角無法完全蓋住」的區域補背景。
5. 按時間距離先近後遠搜尋其他原片幀，排除人物像素，以 ORB 特徵和 RANSAC 單應矩陣對齊背景。必須通過匹配數、內點比例，以及全局和人物附近的顏色一致性檢查，才採用來源像素。
6. 所有候選都無法提供的缺失像素，才用 LaMa 修補。LaMa 即使計算了整張局部圖，程式也只取回缺失遮罩內的預測，不改寫已存在的背景。
7. 將新人物合成回底圖，原片中不屬於編輯遮罩的 RGB 像素保持不變。以 `libx264rgb -crf 0` 無損輸出，保留第一條原音軌（若存在且 MP4 支援該音訊編碼）。

鏡頭運動直接來自原片。幀間配準只用來找背景素材，不會對最終鏡頭做穩定、重構或重新運鏡。

## 安裝

將此包所有檔案按照原目錄結構覆蓋到 MimicMotion 專案根目錄。保留既有 MimicMotion 環境、SVD、MimicMotion 和 DWPose 權重。

另外需要：

- 主環境中可用的 OpenCV、Pillow、NumPy（DWPose 已使用這些影像依賴），以及 PATH 中的 FFmpeg / ffprobe。FFmpeg 需要 `libx264rgb` 編碼器。
- **獨立的 SAM 2 環境**：官方 SAM 2 要求 Python >= 3.10、PyTorch >= 2.5.1 和 torchvision >= 0.20.1；不要直接用它升級專案 `environment.yaml` 中的 PyTorch 2.0.1。
- SAM 2.1 small 權重 `models/sam2.1_hiera_small.pt`。
- 相容 Simple-LaMa 的 TorchScript 權重 `models/big-lama.pt`（模型簽名為 `model(image, mask)`，輸入範圍 0–1；不是原始訓練 checkpoint）。此模型在主環境按需載入。

SAM 2 安裝方式見官方：
https://github.com/facebookresearch/sam2#installation

SAM 2.1 small 權重：
https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt

LaMa 相容 TorchScript 權重來源：
https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt

LaMa 介面參考：
https://github.com/enesmsahin/simple-lama-inpainting/blob/main/simple_lama_inpainting/models/model.py

例如在另一個 conda 環境安裝 SAM 2 後，`--sam2_python` 指向該環境的 Python。下載和安裝應依照以上來源完成；本包不包含模型，執行時也不會偷偷升級主環境。

## 執行

在 MimicMotion 根目錄：

```bash
python inference.py \
  --mode replace \
  --ref_video_path dance.mp4 \
  --ref_image_path new_actor.jpg \
  --sam2_python /path/to/sam2-env/bin/python \
  --sam2_checkpoint models/sam2.1_hiera_small.pt \
  --lama_checkpoint models/big-lama.pt \
  --device cuda:0 \
  --dtype float16 \
  --output_file outputs/replaced.mp4 \
  --keep_intermediates
```

`/path/to/sam2-env/bin/python` 必須改成你的實際路徑。

替換模式不要傳 `--fps`，也不要改用 `--sample_stride 2`；這會改变原片時序，因此程式拒絕此組合。`--conditioning_fps` 仍可調整模型條件，與影片播放幀率不同。

如果採樣幀數加一小於預設 tile size 72，可以使用 `--num_frames 16 --frames_overlap 4`。所有原片幀仍會處理，不會只輸出 16 幀。與原版相同，適用的模型與 tile 設定仍需實際評估。

多人場景須在第一幀框選主角，例如：

```bash
python inference.py \
  --ref_video_path dance.mp4 --ref_image_path new_actor.jpg \
  --person_box 400 120 900 1050 \
  --sam2_python /path/to/sam2-env/bin/python \
  --output_file outputs/replaced.mp4
```

座標為原片解析度中的 `左 上 右 下`。遇到人物追蹤歧義、完全遮擋或找不到姿態，程式會停止，不會默默換掉另一個人。鏡頭切換和多人交錯最好先分成單一連續鏡頭處理。

## 背景補洞控制

| 選項 | 用途 |
| --- | --- |
| `--background_candidates 0` | 預設搜尋所有其他原片幀，找到足夠背景後提早停止；長片可能很慢 |
| `--background_candidates 64` | 限制每幀候選數：先附近，再抽取全片較遠位置。可能更早使用 AI |
| `--no_ai_background` | 禁止 AI 補景；找不到真實素材便失敗並回報缺失像素數 |
| `--mask_padding 3` | 原主角移除區外擴 3 像素，減少殘邊；此邊緣區也屬於允許修改範圍 |
| `--edge_feather 1` | 新人物遮罩向內柔化，避免將生成背景擴到遮罩外；不是專業髮絲級 alpha matting |
| `--mute` | 不帶原音軌 |
| `--keep_intermediates` | 在日誌指定的工作目錄保留追蹤輸入、遮罩與像素來源圖 |

每幀紀錄哪些原片幀貢獻背景、補回多少像素、AI 補了多少像素。JSON 中 `replacement.per_frame` 可查到資料。混合人物邊緣下方也需要背景，所以背景像素計數不是與人物像素互斥的分類。

保留的 `provenance/` 灰階圖：0=當前原片，85=其他原片幀，170=AI 背景，255=生成主角。半透明邊緣優先標示主角；背景來源詳細數量以 JSON 為準。

## 人工遮罩與遮擋

可提供逐幀 PNG：`000000.png`、`000001.png`……，數量必須等於原片幀數，尺寸必須等於原片解析度。

- `--source_masks DIR`：原主角白色，其他黑色。
- `--replacement_masks DIR`：生成主角的 alpha，白色不透明、黑色透明，支援灰階。
- `--occlusion_masks DIR`：白色處永遠保留原片，可保護主角前方的欄杆、道具、其他人等。

同時提供前兩組遮罩時，不需要 SAM 2。遮罩与 `--person_box` 只適用單項任務，避免批次把同一套座標或遮罩套錯影片。人物在真實世界的遮擋層次無法單靠 2D 姿態與人物分割可靠推斷；複雜場景需要精修保護遮罩。

## 限制與驗證範圍

- 生成主角仍是模型預測，無法保證每個關節、手指、表情、衣服和身份完全準確。
- SAM 2 是分割與追蹤工具，無法保證動態模糊和髮絲邊緣完美；錯誤遮罩可能包含生成背景。
- 單應矩陣適合近似平面背景、旋轉和平移等可配準場景。強烈視差、動態人群、水面、燈光變化和切鏡可能無法通過檢查，會增加 AI 使用量。它不是 3D 場景重建。
- LaMa 是逐幀圖像修補，未做生成背景的時間一致性訓練，可能閃爍。AI 補的是合理內容，並不代表原來被遮住的真實背景。
- 原人物的投影、倒影和接觸陰影不一定包含在人物遮罩中，本版沒有自動重建它們。
- 不支援未正規化的 VFR、非方形像素、旋轉標籤或非零影片起始時間；輸入檢查會要求先正規化。
- 無損 RGB H.264 輸出檔案較大，部分播放器不支援；另行轉為一般 YUV420 MP4 會重新壓縮背景像素。
- 原始音訊編碼若不支援 MP4，輸出會報錯；可使用 `--mute`，或先自行轉換原音訊。
- 同時保留所有姿態幀、生成幀和 SAM 2 追蹤狀態需要記憶體。長片全幀生成與跨幀搜尋比之前抽幀生成更慢、更吃記憶體。

已完成：CLI、合成遮罩、原背景像素不變規則、平移鏡頭下的真實背景回填、不相干場景拒絕、無損編解碼、以及用模擬 AI 檢查僅在缺失區域補景。這些檢查不等於已驗證 MimicMotion / SAM 2 / LaMa 的真實模型效果；交付環境沒有這些權重與完整 GPU 推理依賴。


## Dual T4 component placement

Use `--device cuda:0 --aux_device cuda:1 --dtype float16` to place UNet on GPU 0 and PoseNet, image encoder, VAE, and replacement helpers on GPU 1. Decode chunks default to 2 in this mode. See [DUAL_T4.md](DUAL_T4.md) for commands, limitations, and validation status.
