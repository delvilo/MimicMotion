# 完整安裝（main：單 GPU）

雙 GPU 程式保留在 `codex/dual-t4-component-split` 分支。本安裝腳本不切換分支、不合併 PR，也不把雙 GPU 功能加入 main。

## 一次安裝

適用 Linux x86_64，建議 Ubuntu 22.04／24.04 或 Ubuntu WSL2；使用者須已有可正常運作的 NVIDIA 驅動。腳本不安装、更換或升級顯示卡驅動，也不要求本機安裝 CUDA toolkit / nvcc。

```bash
git clone --branch main https://github.com/delvilo/MimicMotion.git
cd MimicMotion
bash scripts/install.sh
```

已經有倉庫時，先在 `main` 拉取最新版，再執行安裝腳本。腳本不替使用者切換分支或處理未提交程式修改。

**SVD 模型需要帳號存取權。** 請先在瀏覽器登入 Hugging Face，依模型頁面申請／同意所需條件：

https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1

下載時使用既有的 Hugging Face 登入，或在執行腳本前安全輸入讀取 token：

```bash
read -rsp 'Hugging Face token: ' HF_TOKEN; echo
export HF_TOKEN
bash scripts/install.sh
unset HF_TOKEN
```

Token 只透過環境傳遞，不寫入安裝腳本、啟動器或執行紀錄。腳本不會替你接受模型條款；未取得權限時會停止並指出原因。網路錯誤或中途停止後可以重跑；已下載的 Hugging Face 檔案可重用快取。

## 腳本安裝內容

1. 透過 apt 安裝 FFmpeg、curl、git、系統影像函式庫等。非 root 使用 sudo，可能需要輸入系統密碼。
2. 在 `.runtime/` 安裝固定版 uv，驗證官方下載 checksum，並建立受管理 Python 3.11。
3. 建立兩個獨立環境，不升級系統 Python 或原本 conda 環境：

| 環境 | 版本解析方式 |
| --- | --- |
| `.runtime/envs/cuXXX/main` | 依所選 CUDA wheel 來源解析 PyTorch 與 torchvision 的相容版本；CUDA runtime 必須 > 11.7 |
| `.runtime/envs/cuXXX/sam2` | PyTorch >= 2.5.1、相容 torchvision；固定 SAM 2 程式碼版本，與主環境隔離 |

不再固定 `torch==2.0.1`、`torch==2.5.1` 或 CUDA 11.8。預設讀取 NVIDIA 驅動可支援的 CUDA 上限，查詢官方 wheel 目錄，選擇目前支援範圍內的最新 CUDA family；再由 pip 解析該來源上的套件版本。不同 CUDA family 使用不同子目錄，保留舊環境，避免混用 CUDA 函式庫。

也可明確指定官方來源，**但不指定 PyTorch 版本**：

```bash
bash scripts/install.sh --torch-index-url https://download.pytorch.org/whl/cu126
```

`cu126` 只是選擇 CUDA 12.6 wheel 來源的例子，並非預設或硬編碼要求。無 GPU 主機使用 `--skip-gpu-check` 時，需要明確提供目標主機的 wheel 來源；不能從不存在的驅動推斷目標版本。

ONNX Runtime 依實際 `torch.version.cuda` 和 `torch.backends.cudnn.version()` 選擇 ABI 相容範圍：

| CUDA／cuDNN | 主環境解析條件 |
| --- | --- |
| CUDA 11.8／cuDNN 8 | PyTorch >= 2.0、< 2.4；ONNX Runtime >= 1.16、< 1.19 |
| CUDA 12.x／cuDNN 9 | PyTorch >= 2.4；ONNX Runtime >= 1.19、< 1.27 |
| CUDA 13.x／cuDNN 9 | PyTorch >= 2.4；ONNX Runtime >= 1.27、< 1.30 |

這些是已知 wheel 的相容區間，不是單一版本鎖定。CUDA 11.8 主環境的 PyTorch 上限是 cuDNN 8 ABI 要求；SAM 2 的獨立環境仍使用 >= 2.5.1。未列出的 CUDA/cuDNN 組合會明確拒絕，避免裝好後偷偷退回 CPU。將來出現新的 CUDA family 或 ONNX Runtime ABI 時，需要更新此相容性表，不能保證任意未來版本都可用。


4. 安裝 `scripts/requirements-main.txt` 與 `scripts/requirements-sam2.txt`，檢查套件相依關係。SAM 2 的可選 CUDA extension 關閉，因此不需要 nvcc；官方說明指出部分後處理功能可能受限。
5. 下載 DWPose 的兩個 ONNX、MimicMotion 1.1、SVD 所需的 FP16 VAE／image encoder 與設定、SAM 2.1 small、LaMa TorchScript。
6. 檢查主程式依賴匯入、PyTorch FP16 CUDA 運算、SAM 2 CUDA、DWPose session 與模型載入。LaMa 會做小尺寸測試；SAM 2 權重會在 CPU 載入檢查。
7. 產生啟動器與套件版本清單；Hugging Face revision 及模型本地 SHA-256 記在 `models/install-manifest.json`。

SVD 只下載本程式實際使用的檔案：UNet 由設定建立並載入 MimicMotion 權重，因此不另外下載 SVD UNet 全部權重。

預留足夠磁碟空間；雙環境、pip/Hugging Face 快取、模型、生成影片會使用數十 GB。驗證會使用主記憶體，不是只檢查檔名。當磁碟、記憶體、GPU、驅動或模型授權不符時會停止，不會聲稱安裝成功。

## 啟動單 GPU 人物替換

```bash
.runtime/bin/run-mimicmotion \
  --ref_video_path /absolute/path/dance.mp4 \
  --ref_image_path /absolute/path/new_actor.jpg \
  --device cuda:0 \
  --dtype float16 \
  --output_file outputs/replaced.mp4
```

啟動器會指定主環境、SAM 2 專用啟動器及本地模型路徑；不必手動 activate 環境。執行時會切回專案根目錄，所以素材若不在專案內，請使用絕對路徑。

兩個環境可能使用不同的 cuDNN 主版本，SAM 2 啟動器會移除主環境注入的動態函式庫路徑，避免互相干擾。

原本整幅生成可加 `--mode generate`。main 不接受 `--aux_device`；不要在此單 GPU 命令加上雙卡參數。

## 選項與重新執行

```bash
bash scripts/install.sh --help
bash scripts/install.sh --dry-run
bash scripts/install.sh --prefix /data/mimicmotion-runtime
bash scripts/install.sh --skip-system-deps
```

- `--dry-run`：顯示計畫，不改檔案、不下載、不呼叫 apt。
- `--torch-index-url URL`：選擇官方 CUDA wheel 來源；省略時依驅動自動選擇，不固定 PyTorch 發行版本。
- `--prefix DIR`：改變環境、快取、啟動器與安裝紀錄位置；模型仍在專案 `models/`。啟動器改用 `DIR/bin/run-mimicmotion`。
- `--skip-system-deps`：系統套件已安裝、或沒有 apt／sudo 時使用。缺少 FFmpeg 等必要命令仍會報錯。
- `--skip-models`：只安裝環境；之後不帶此選項重跑才能完成模型準備。
- `--skip-gpu-check`：可在無 GPU 主機預先準備，略過 GPU 檢查；不表示 GPU 已經可用。之後在 GPU 主機不帶此選項重跑。

重跑不會清空環境或覆蓋外部自行建立的環境。如果指定位置已有不屬於此腳本的環境，腳本會拒絕修改。安裝以檔案鎖防止同一 prefix 並行執行。

模型若已存在但與下載來源不同，會要求使用者先明確處理，不會默默覆蓋自訂的 MimicMotion／DWPose 權重。SVD 目錄由 Hugging Face snapshot 管理，請勿拿來保存自訂權重。

## 驗證範圍

交付前檢查了 Bash 語法、參數／dry-run 行為、Python 輔助程式編譯與安裝資源來源。安裝腳本會在使用者主機實際驗證環境和模型。

開發環境未執行完整 CUDA 安裝與所有模型下載，也尚未以真實舞蹈片完成生成驗收。安裝完成的 smoke checks 不代表動作、身份、分割邊緣與背景修補品質已驗證。

相容性來源：
- https://pytorch.org/get-started/previous-versions/
- https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
- https://github.com/facebookresearch/sam2#installation


## 使用 Conda 的版本範圍

`environment.yaml` 現在使用 `pytorch`、`torchvision`、`pytorch-cuda>11.7`，由 conda 解析來源中可用的版本；不再指定 PyTorch 2.0.1 或 CUDA 11.7。Conda channel 可提供的版本與 pip wheel 不一定相同，完整安裝仍建議使用上述 Bash 腳本。

自行建立 conda 環境時，可以在安裝主依賴後配置相容的 ONNX Runtime：

```bash
python -m pip install -r scripts/requirements-main.txt
python scripts/cuda_runtime.py --install-ort
```

`--install-ort` 只根據已安裝的 PyTorch 配置 ONNX Runtime，不會擅自替換 PyTorch。若 conda 解出 CUDA 11.8／cuDNN 9 等未支援組合，它會報錯；可改用腳本選擇相容版本範圍。

已測試選擇邏輯與拒絕條件；尚未在所有 CUDA family 上安裝並執行完整模型。實際套件版本會寫入 `.runtime/reports/`，方便重現與排錯。升級版 PyTorch 是否能搭配所有既有模型 API，仍須以安裝器的 import／CUDA／模型檢查及實片推理確認。
