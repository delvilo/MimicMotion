# 透明背景影片使用說明

> **專案說明：** 本專案 fork 自 [Tencent/MimicMotion](https://github.com/Tencent/MimicMotion)；本 fork 新增的功能由 ChatGPT 完成。

透明背景模式會只輸出生成的人物，並為人物以外的區域寫入透明 Alpha。它適合把人物放到剪輯軟體中的其他背景上。這個功能已加入 `main` 與 `codex/dual-t4-component-split` 分支。

## 快速開始

在專案根目錄執行：

```bash
python inference.py \
  --ref_video_path dance.mp4 \
  --ref_image_path actor.png \
  --transparent_background \
  --alpha_codec prores4444 \
  --output_file outputs/actor.mov
```

`--transparent_background` 開啟透明輸出。透明影片必須使用 `.mov` 副檔名；若省略 `--output_file`，程式會在 `--output_dir` 中自動建立 `.mov` 檔案。沒有開啟透明模式時，原有輸出方式維持不變。

預設模式為 `--mode replace`：使用舞蹈影片的姿態生成動作，保留原鏡頭的畫面尺寸、影格數、幀率與人物座標，再將生成的人物輸出成透明影片。若指定 `--mute`，會省略原影片音訊；否則若原影片有音訊，第一條音軌會轉成 PCM 並放入 MOV 檔案。

也可使用 `--mode generate --transparent_background`。這個模式沿用一般生成模式的畫面尺寸、抽幀規則與 `--fps`，不帶入原影片音訊。若畫面中有多個人物，或人物追蹤不明確，請提供 `--person_box` 指定主角，或提供人工修正過的遮罩。

## 編碼器選擇

| 參數 | 說明 |
| --- | --- |
| `--alpha_codec prores4444` | 預設。ProRes 4444，可保留 Alpha，適合後製剪輯；RGB 畫面採有損壓縮。 |
| `--alpha_codec qtrle` | 無損 RGBA 編碼，檔案通常較大。 |

兩種格式均封裝在 MOV 中。一般 H.264 MP4 不適用於此功能。部分播放器會把透明區顯示成黑色；請將影片放到剪輯軟體的背景圖層上，確認透明效果。輸出的 Alpha 是 straight/unassociated alpha；若軟體提供 Alpha 解讀選項，請選擇 straight 或 unassociated。

## 遮罩與人物選擇

預設會使用 SAM 2 從生成影格分割並追蹤人物。請準備 SAM 2 checkpoint 及可執行 SAM 2 的 Python 環境；若已使用安裝腳本建立獨立環境，請沿用安裝時設定的 `--sam2_python` 和模型路徑。透明模式不使用 LaMa。

若你已經逐幀修正人物遮罩，可用 `--replacement_masks masks/` 指定遮罩資料夾。提供完整遮罩後，程式不需要 SAM 2。遮罩檔名須依序為 `000000.png`、`000001.png` 等，每個輸出影格各一張；每張都是與輸出畫面同尺寸的灰階 PNG：黑色代表透明、白色代表不透明、灰階代表半透明。缺少檔案、尺寸不合或整張遮罩全黑時，程式會停止輸出。

`--edge_feather 1` 會向人物遮罩內側柔化約 1 像素的邊緣；設為 `--edge_feather 0` 可保留輸入遮罩邊緣。`--person_box X0 Y0 X1 Y1` 用來指定首幀人物的矩形範圍，座標格式為左上角 X、Y 及右下角 X、Y。在 `replace` 模式中，座標以原舞蹈影片的像素為準；在透明背景的 `generate` 模式中，座標以生成畫面像素為準。

`--keep_intermediates` 會保留 SAM 2 遮罩、提示框與相關中間資料，方便檢查或調整。遮罩選項只支援一次處理一段影片。

透明模式只使用 `--replacement_masks`。`--source_masks` 和 `--occlusion_masks` 是原背景合成用的參數，不可與透明模式併用。`--mask_padding`、`--background_candidates`、`--no_ai_background` 也只作用於背景合成，在透明模式中不生效。

## 雙 GPU 分支

在 `codex/dual-t4-component-split` 分支，透明輸出命令可加入：

```bash
--device cuda:0 --aux_device cuda:1 --dtype float16 --decode_chunk_size 2
```

UNet 位於第一張 GPU；PoseNet、影像編碼器、VAE 與 SAM 2 位於第二張 GPU。遮罩人工輸入、編碼器及透明背景參數與 `main` 相同。雙 GPU 的裝置參數不適用於 `main` 分支。更多雙 GPU 安裝與記憶體限制請見 [雙 T4 元件分卡說明](DUAL_T4.md)。

## 目前限制

`replace` 模式沿用原背景替換流程的來源影片條件，例如可變幀率影片須先轉成固定幀率。`generate` 模式的時長與輸出幀率依既有生成參數決定。

SAM 2 產生的是人物分割遮罩，並非專門處理髮絲、薄紗等細節的 matting 模型。快速動作、細髮、半透明材質或人物互相遮擋時，邊緣可能不完整，建議檢查影片或提供人工修正的 `--replacement_masks`。此功能不會生成被遮擋的人體部分，也不保證每一影格都能完美去背。

透明背景模式不會搬用或修補原影片背景，也不會用 AI 繪製新背景；人物以外的區域會保持透明。若你需要保留舞蹈影片背景，請使用既有的背景替換模式，不要開啟 `--transparent_background`。

## CPU 回歸測試

在已安裝 NumPy、Pillow 與 FFmpeg/ffprobe 的環境中，可執行：

```bash
python -m unittest discover -s tests -v
```

測試會驗證 ProRes 4444 與 QTRLE 編碼後的 Alpha、分數幀率、命令列參數、遮罩輸入路徑，以及雙 GPU 分支的 SAM 2 裝置路由。這些測試不包含模型權重與實體 GPU，因此不取代完整 GPU 端到端測試。

另有[功能摘要](TRANSPARENT_OUTPUT.md)。
