# 透明背景影片 / Transparent actor output

加入 `--transparent_background`，只輸出生成角色及 Alpha 通道。此選項在 `main` 與 `codex/dual-t4-component-split` 都可使用；未指定時，仍依原本模式輸出一般 MP4。

```bash
python inference.py \
  --ref_video_path dance.mp4 \
  --ref_image_path actor.png \
  --transparent_background \
  --alpha_codec prores4444 \
  --output_file outputs/actor.mov
```

若使用安裝腳本建立的獨立 SAM 2 環境，請沿用安裝後提供的 `--sam2_python` 路徑與模型路徑。透明輸出需要 SAM 2 及其 checkpoint；若提供完整的 `--replacement_masks`，則不需要 SAM 2。兩者都不需要 LaMa。

雙 GPU 分支在上述命令加入以下參數即可；這些裝置參數不適用於 `main`：

```bash
--device cuda:0 --aux_device cuda:1 --dtype float16 --decode_chunk_size 2
```

雙 GPU 模式維持 UNet 在第一張 GPU，PoseNet、影像編碼器、VAE 與 SAM 2 在第二張 GPU。透明輸出不執行背景修補。

| 參數 | 行為 |
| --- | --- |
| `--transparent_background` | 開啟透明背景 MOV 輸出；省略即維持一般輸出 |
| `--alpha_codec prores4444` | 預設；ProRes 4444，保留 Alpha，RGB 為有損編碼，適合剪輯 |
| `--alpha_codec qtrle` | QTRLE 無損 RGBA；檔案可能很大 |
| `--output_file outputs/actor.mov` | 透明模式必須是 `.mov`；不指定時自動以 `.mov` 命名 |
| `--edge_feather 1` | 向內柔化遮罩邊緣的像素半徑；`0` 保留原遮罩 Alpha |
| `--replacement_masks masks/` | 使用人工修正的逐幀灰階 PNG；黑色透明、白色不透明、灰色半透明 |
| `--person_box X0 Y0 X1 Y1` | 指定主角：replace 使用原影片首幀座標；透明 generate 使用生成影片首幀座標 |
| `--keep_intermediates` | 保留 SAM 2 遮罩、提示框及編碼紀錄，便於修正 |
| `--mute` | replace 透明模式不帶入原影片音訊 |

遮罩依序命名 `000000.png`、`000001.png` 等，每個輸出影格一張，尺寸須與輸出一致；只支援單一輸入任务。錯誤尺寸或空主角遮罩會停止輸出。透明模式不接受 `--source_masks` 或 `--occlusion_masks`，它們只用於原背景合成。背景搜尋、移除遮罩擴張及 `--no_ai_background` 在透明模式沒有作用。

預設 `--mode replace` 保留原影片尺寸、影格數、固定幀率與主角所在的鏡頭座標；只把畫面背景設成透明。原影片第一條音訊（若存在）會轉為 PCM 寫入 MOV。既有來源影片限制仍適用，例如可變幀率需先轉為固定幀率。

也支援 `--mode generate --transparent_background`：使用原有生成畫幅、取樣規則與 `--fps`，不帶入原音訊。未提供遮罩時，會從生成影格偵測主角，再交給 SAM 2 追蹤；多主角或追蹤不明確時會要求指定主角或修正素材。

透明模式完全跳過原背景重建、跨幀補洞及 LaMa。輸出是 straight/unassociated Alpha，剪輯軟體請選擇相應的 Alpha 解讀方式。一般播放器可能把透明區顯示成黑色；應放在剪輯軟體另一層背景上確認，不代表檔案缺少 Alpha。一般 MP4/H.264 不作為此功能的透明格式。

SAM 2 提供分割遮罩，不是專門的髮絲或半透明材質 matting 模型；快速動作、髮絲、薄紗與遮擋可能需要人工修正 `--replacement_masks`。此功能不保證完美去背或恢復原本被遮擋的身體部位。

CPU 回歸驗證（需 NumPy、Pillow、FFmpeg、ffprobe）：

```bash
python -m unittest discover -s tests -v
```

測試包含兩種編碼的 Alpha 解碼驗證、分數幀率、參數驗證、略過背景/LaMa/SAM 的遮罩輸入路徑，以及 SAM 2 裝置路由。不取代帶模型權重的 GPU 端到端驗證。
