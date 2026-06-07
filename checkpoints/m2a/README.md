# M2A accompaniment checkpoint

대용량 가중치라 git이 아닌 **GitHub Releases**로 배포합니다.

1. 이 repo의 **Releases**(`v1.0`) 에서 `best-epoch.007-val_loss.0.8431.ckpt` 를 내려받아
2. 이 폴더(`checkpoints/m2a/`)에 둡니다.

```bash
gh release download v1.0 -R stemkwk/melody-miner -p "best-epoch*.ckpt" -D checkpoints/m2a
```

`python app.py` / `run.py` 시작 시 `checkpoints/m2a/` 의 `.ckpt` 를 자동 인식합니다.
