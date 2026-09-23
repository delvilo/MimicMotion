# 兩張 T4：元件分卡模式（方案 D）

> **專案說明：** 本專案 fork 自 [Tencent/MimicMotion](https://github.com/Tencent/MimicMotion)；本 fork 新增的功能由 ChatGPT 完成。

此版本將同一支影片的模型元件放在兩張 CUDA GPU。不是兩支影片各跑一張卡，也沒有把 UNet 的網路層拆到兩卡。

| 元件／資料 | 位置 | 處理階段 |
| --- | --- | --- |
| UNet、scheduler、去噪 latent、隨機數 generator | GPU 0 | 人物生成 |
| PoseNet | GPU 1 | 姿態條件運算；輸出搬到 GPU 0 給 UNet |
| Image encoder | GPU 1 | 參考圖編碼；結果搬到 GPU 0 |
| VAE encoder／decoder | GPU 1 | 編碼與分段解碼 |
| DWPose、SAM 2、LaMa | GPU 1 | 前處理／後處理；按原流程先後執行 |
| 完整解碼前 latent 與已解碼畫面 | 主記憶體 | 解碼期間逐段搬進 GPU 1 |

使用 `--device cuda:0 --aux_device cuda:1 --dtype float16` 啟用。兩卡必須在同一個程序中可見。雙卡模式要求生成模型使用 FP16；沒有使用 T4 不適合的 BF16 預設。SAM 2 與 LaMa 保持其現有推理精度，並非所有輔助模型都強制轉為 FP16。

## 執行人物替換

先依 `REPLACEMENT.md` 準備模型與獨立 SAM 2 環境。在專案根目錄執行：

```bash
CUDA_VISIBLE_DEVICES=0,1 python inference.py \
  --mode replace \
  --ref_video_path dance.mp4 \
  --ref_image_path new_actor.jpg \
  --device cuda:0 \
  --aux_device cuda:1 \
  --dtype float16 \
  --decode_chunk_size 2 \
  --sam2_python /path/to/sam2-env/bin/python \
  --sam2_checkpoint models/sam2.1_hiera_small.pt \
  --lama_checkpoint models/big-lama.pt \
  --output_file outputs/replaced_dual_t4.mp4
```

替換 Python、圖片與影片路徑。SAM 2 子程序會繼承 `CUDA_VISIBLE_DEVICES`，並使用相同的邏輯 `cuda:1`；不要在 SAM 2 的啟動器中另行改寫可見 GPU。

若使用 `CUDA_VISIBLE_DEVICES=2,3`，程式裡仍用 `cuda:0`、`cuda:1`，分別對應實體卡 2、3。

## 原本整幅生成模式

```bash
CUDA_VISIBLE_DEVICES=0,1 python inference.py \
  --mode generate \
  --ref_video_path dance.mp4 --ref_image_path new_actor.jpg \
  --device cuda:0 --aux_device cuda:1 --dtype float16 \
  --decode_chunk_size 2 --output_dir outputs/generated
```

不傳 `--aux_device` 就是單卡模式。沒有指定 `--decode_chunk_size` 時，雙卡預設 2、單卡保留 8。JSON 參數紀錄包含兩張卡的邏輯裝置名稱。

## 效果與限制

- 降低 GPU 0 同時承擔姿態運算與 UNet 的顯存需求，並將 VAE 解碼移到 GPU 1。
- 兩張 T4 的顯存不會合成一個連續的 32GB 池；UNet 和去噪 latent 仍必須放得進 GPU 0。
- 這是元件分卡，不保證雙卡負載均衡。GPU 1 可能在 UNet 運算期間等待。
- 有跨卡資料傳輸與主記憶體搬運，可能更慢；沒有實測加速倍數。
- 長影片、高解析度、大 tile 仍可能 OOM。GPU 0 OOM 時，先降低 `--resolution`（64 的倍數）或評估較小的 `--num_frames` 和相應 `--frames_overlap`；只降低解碼 chunk 不會解決 UNet 的 OOM。
- GPU 1 在 VAE 解碼 OOM 時可以用 `--decode_chunk_size 1`；這不控制 SAM 2 或 LaMa 的顯存使用。SAM 2／LaMa 因全解析度或長片 OOM 時，需要縮短或另行降低輸入尺寸，不能靠這個參數解決。
- 降低解碼 chunk 可能影響時間解碼品質；降低解析度／tile 也可能影響輸出細節和動作連貫性，需用片段比較。
- 主記憶體仍需容納影片、姿態、生成畫面及追蹤資料。
- 原背景優先回填、AI 僅補未找到的缺口，以及人物遮罩合成流程均保留。

## 驗證

已完成 Python 編譯、CLI 單／雙卡預設、拒絕相同／不存在／CPU 輔助裝置、FP16 約束，以及對實際解碼函式的模擬張量測試（5 幀按 2、2、1 搬到 VAE 所在裝置）。已檢查單卡既有流程。

尚未在兩張實體 T4 和完整權重上跑過模型；跨卡 PyTorch 運算、峰值顯存、輸出品質與執行時間仍需硬體驗收。

## 透明背景輸出

雙 GPU 分支支援透明背景 `.mov` 輸出，SAM 2 會放在 GPU 1。請參閱[繁體中文透明影片說明](TRANSPARENT_OUTPUT.zh-TW.md)，其中列有遮罩格式、參數、編碼器與限制。
