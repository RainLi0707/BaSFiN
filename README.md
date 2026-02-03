# BaSFiN: Bayesian Skill Updating for Sports Matchup Prediction

## 📖 專案簡介
本研究提出一套專為 **體育對決預測任務** 設計的深度學習架構 **BaSFiN**。  
核心結合以下三大要素：

- **貝式技能更新機制 (Bayesian Skill Updating)**：透過貝氏後驗更新動態追蹤選手能力的浮動與不確定性。  
- **特徵交互建模 (Feature Interaction Modeling)**：結合指數移動平均（EMA）特徵，有效捕捉 **非遞移性效應**，提升模型對賽局判斷能力。  
- **時序特徵處理 (Temporal Feature Processing)**：強化對選手能力變化的表徵能力。  

此外，透過 **預訓練凍結骨幹網路 (frozen backbone)**，先獲取穩定表徵，再進行整合層微調，提升模型的 **穩健性與泛化能力**。  
實驗結果顯示，本方法在多個體育競技資料集上顯著優於傳統對決模型，展現了 **貝式推論與深度神經網路整合的潛力**。  

---

## 📂 專案結構總覽
```
BaSFiN_code/
│
├── code/                 # 主要模型與訓練程式碼
│   ├── BaSFiN/           # 模組主體
│   ├── logs/            # 訓練與實驗紀錄
│   └── model/           # 預訓練與訓練後模型儲存
│
├── data/                 # 資料與特徵
│   ├── ema_tensor/       # 特徵交互模型用 tensor 與 mapping
│   │   ├── ematensor.pt
│   │   └── game_id_mapping.json
│   ├── feature_csv/      # 隨比賽場次選手特徵統計表
│   ├── final_data/       # 對決資料表
│   └── player_id_mapping_2009_2024.csv  # 玩家編號與名稱對應
│
├── output/              # 模型輸出結果與推論結果
│
└── requirements.txt     # 套件環境安裝需求
```

---

## 🧾 資料說明

### `data/player_id_mapping_2009_2024.csv`
- 儲存選手的 ID 與對應名稱，用於還原結果中球員資訊。

### `data/ema_tensor/ematensor.pt`
- 為特徵交互模型所用的 **指數移動平均（EMA）張量特徵**，包含動態時間序列資訊。

### `data/ema_tensor/game_id_mapping.json`
- 對應每場比賽的 game_id 至資料序列，供模型讀取 `ematensor.pt`。

### `data/feature_csv/`、`final_data/`
- 最終統一格式的數據結構。

### `data/data_2013_2024.csv`（範例格式如下）：
| year | id   | player1 | ... | player10 | target |
|------|------|---------|-----|----------|--------|
| 2013 | 5013 | 256     | ... | 443      | 1      |

- 每筆資料代表一場對局：
  - `player1`～`player5` 為主隊先發
  - `player6`～`player10` 為客隊先發
  - `target` 為對局結果（1 = 主隊勝，0 = 客隊勝）

---

## 🚀 使用方式

### 1. 安裝套件依賴
```bash
pip install -r requirements.txt
```

### 2. 預訓練子模組（依序執行）
```bash
python code/pretrain.py     #預先訓練儲存子模型最佳優化
python code/train_BaS.py    #單獨執行觀察統計特徵結果
python code/train_bc.py     #單獨執行觀察統計特徵結果
python code/train_cofim.py  #單獨執行觀察統計特徵結果

```

### 3. 執行整合訓練（BaSFiN）
```bash
python code/train_BaSFiN.py
```

---

## 🧪 研究貢獻
- ✅ 結合貝式推論與深度神經網路，實現動態技能更新與不確定性追蹤  
- ✅ 捕捉非遞移性效應結合時序特徵平滑，提升對複雜賽局的判斷能力  
- ✅ 採用預訓練凍結骨幹策略，提高模型穩健性與泛化能力  

---

## 🔑 關鍵詞
深度學習、貝式定理、神經網路、非遞移性、對決預測
