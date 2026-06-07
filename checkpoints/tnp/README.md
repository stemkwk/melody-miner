# TNP voice-conversion checkpoint

대용량 가중치라 git이 아닌 **GitHub Releases**로 배포합니다.

1. 이 repo의 **Releases** 탭에서 `latest.pt` 를 내려받아
2. 이 폴더(`checkpoints/tnp/`)에 둡니다.

```bash
gh release download -R <owner>/melody-miner -p "latest.pt" -D checkpoints/tnp
```

이 파일이 없으면 파이프라인은 **음성 변환을 건너뛰고 입력 WAV를 보컬 스템으로
사용**합니다(Branch A 반주 생성은 정상 동작). 준비되면:

```bash
python run.py --input song.wav \
    --m2a-checkpoint "checkpoints/m2a/best-epoch=007-val_loss=0.8431.ckpt" \
    --tnp-checkpoint checkpoints/tnp/latest.pt \
    --reference references/target1.wav references/target2.wav \
    --out output/run
```
