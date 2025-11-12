# Numbers3 — Clean Start

最小構成の**再設計版**です。`src/n3` に機能を分離し、CLI一発で **学習→次回予測→期待値** まで回ります。

## クイックスタート

```bash
cd numbers3_clean
python -m venv .venv && . .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

# サンプルデータで動作確認
python -m n3.cli --history data/sample/history.csv --run all

# 生成物
# - artifacts/models/model.pkl
# - artifacts/outputs/next_prediction.csv
# - artifacts/outputs/ev_report.csv
```
