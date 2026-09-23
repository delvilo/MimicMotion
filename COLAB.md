# Google Colab 安裝與執行（main 單 GPU）

> **專案說明：** 本專案 fork 自 [Tencent/MimicMotion](https://github.com/Tencent/MimicMotion)；本 fork 新增的功能由 ChatGPT 完成。

[在 Colab 開啟可執行筆記本](https://colab.research.google.com/github/delvilo/MimicMotion/blob/main/MimicMotion_Colab.ipynb)。依序執行各儲存格，筆記本會安裝環境、檢查輸入、執行影片並將結果複製到 Google Drive（可關閉）。本頁也提供一個 Bash 命令的操作方式。

開始前，在 Colab「執行階段 → 變更執行階段類型」選擇 **GPU** 並重新連線。Google 動態分配 GPU 與使用時間，實際硬體和容量可能不同，詳見 [Colab 官方說明](https://research.google.com/colaboratory/intl/zh-TW/faq.html)。建議讓輸入、模型與生成檔案有足夠的 `/content` 暫存空間；執行階段重設後，`/content` 資料可能消失，請將完成的影片複製到 Drive。

須先登入 Hugging Face，在瀏覽器取得 [SVD 模型](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1)的存取權，並準備有讀取權限的 token。筆記本優先讀取 Colab Secrets 中名為 `HF_TOKEN` 的項目，否則以隱藏輸入框讀取；腳本透過環境變數使用 token，不把 token 寫進文件或命令列參數。請不要將 token 寫入公開筆記本。

## Colab 操作步驟

1. 按上方連結開啟筆記本，選好 GPU 執行階段。
2. 執行 Git clone 與 GPU 檢查儲存格，提供 Hugging Face token。
3. 執行安裝儲存格。它呼叫 `scripts/colab.sh --install-only`，進而使用現有的 `scripts/install.sh`：建立獨立主環境和 SAM 2 環境、下載並驗證模型，以及產生 `.runtime/bin/run-mimicmotion`。依 Colab GPU 驅動選擇適合的 CUDA wheel；不在腳本中固定 PyTorch 版本。
4. 可掛載 Google Drive，填入舞蹈影片和角色圖片的**完整路徑**；也可用 Colab 左側的檔案面板上傳至 `/content`。筆記本將輸入複製到 `/content` 本地磁碟，避免在 Drive 檔案系統直接寫入輸出影片。
5. 設定 `MODE`、`TRANSPARENT_BACKGROUND` 等欄位，執行推論儲存格。輸出會先保存在 `/content/MimicMotion/outputs/colab/`，之後再複製到 Drive（若已選擇）。

預設 `MODE=replace`，保留舞蹈影片原鏡頭和背景，讓生成角色替換主角。設為 `TRANSPARENT_BACKGROUND=True` 則只輸出有 Alpha 通道的人物 `.mov`；可用 `ALPHA_CODEC` 選擇 ProRes 4444 或無損 QTRLE。`MODE=generate` 使用既有的整幅生成流程。詳細參數與去背限制請參閱[繁體中文透明影片說明](TRANSPARENT_OUTPUT.zh-TW.md)。

筆記本示範 `--resolution 384 --num_frames 16 --frames_overlap 4 --decode_chunk_size 2` 作為較低記憶體的初始設定，畫質與時長不由這些數值保證；GPU 顯存不足時可再降低解析度或縮短影片。`num_frames` 是模型 tile 的長度，不是最終影片的總影格數。`replace` 模式需要固定幀率來源，並會保留來源的影格數。

## 直接在 Colab 儲存格使用 Bash 腳本

以下 Bash 區塊可貼進 Colab 程式儲存格直接執行。先 clone 主分支；輸入素材請自行上傳到 `/content`，或掛載 Drive 並在推論前複製到 `/content`：

```bash
%%bash
git clone --depth 1 --branch main https://github.com/delvilo/MimicMotion.git /content/MimicMotion
```

可在 Colab Secrets 儲存 `HF_TOKEN` 並使用筆記本的隱藏輸入儲存格；使用 Bash 時請用隱藏輸入設定環境變數，並確認已取得 SVD 存取權。安裝命令：

```bash
%%bash
bash /content/MimicMotion/scripts/colab.sh --install-only
```

第一次推論：

```bash
%%bash
bash /content/MimicMotion/scripts/colab.sh --run-only -- \
  --ref_video_path /content/dance.mp4 \
  --ref_image_path /content/actor.jpg \
  --mode replace --device cuda:0 --dtype float16 \
  --resolution 384 --num_frames 16 --frames_overlap 4 --decode_chunk_size 2 \
  --output_file /content/MimicMotion/outputs/colab/result.mp4
```

省略 `--run-only` 時，腳本會先執行安裝器再推論，適合只跑一次的一行命令。要輸出透明影片，加入 `--transparent_background --alpha_codec prores4444`，並把輸出副檔名改為 `.mov`。需要重跑同一路徑時加入 `--overwrite`。`--dry-run` 可顯示安裝計畫和推論命令，不下載模型也不需要 GPU。

腳本只針對 `main` 的單 GPU 啟動器。雙 GPU 版本保留在獨立分支，請參閱該分支的 `DUAL_T4.md`。Colab 無法保證會分配兩張 GPU。

安裝器會檢查 FFmpeg、GPU、PyTorch FP16、ONNX、模型權重和 SAM 2；本專案交付環境未取得 Colab GPU 或 SVD 授權，尚未完成真實影片的 Colab 端到端驗證。若執行中遇到錯誤，先檢查安裝儲存格的原始錯誤，再確認模型授權、磁碟空間與所分配 GPU 的記憶體。
