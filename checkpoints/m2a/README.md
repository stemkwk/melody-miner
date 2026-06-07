# M2A accompaniment checkpoint

대용량 가중치라 git이 아닌 **GitHub Releases**로 배포합니다.

1. 이 repo의 **Releases** 탭에서 `best-epoch=007-val_loss=0.8431.ckpt` 를 내려받아
2. 이 폴더(`checkpoints/m2a/`)에 둡니다.

```bash
gh release download -R <owner>/melody-miner -p "best-epoch=007*.ckpt" -D checkpoints/m2a
```

체크포인트는 `python app.py` / `run.py` 시작 시 `checkpoints/m2a/` 에서 자동 인식됩니다.
