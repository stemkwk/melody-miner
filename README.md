# melody-miner

하나의 **입력 WAV(노래/허밍)** 에서 두 갈래를 돌려 한 곡으로 합칩니다.

```
                 ┌─ Branch A ─ WAV → MIDI 전사 → M2A Transformer 반주 생성 → 반주 WAV ─┐
입력 WAV ──┤                                                                          ├─ 믹스 → 05_final_mix.wav
                 └─ Branch B ─ WAV → TNP voice conversion(목표 화자) → 변환 보컬 WAV ───┘
```

두 브랜치는 **같은 입력 WAV에서 파생**되므로 입력의 t=0 타임라인을 공유합니다 →
시간 워핑 없이 샘플레이트 정렬 + 길이 패딩만으로 동기화되어 믹스됩니다.

- **Branch A** = melody-to-accompaniment-transformer (M2A) — [`src/m2a_transformer/`](src/m2a_transformer/README.md)
- **Branch B** = 음성 변환 모델 — [`src/tnp_voice_conversion/`](src/tnp_voice_conversion/README.md)

> 두 원본 프로젝트는 **소스 복사(vendoring)** 되어 있고 이 repo는 둘을 오케스트레이션만 합니다.

각 모델의 아키텍처·학습·추론 세부 사항은 위 링크를 참조하세요. 이 README는 두 브랜치를 잇는 **오케스트레이션 레이어**만 다룹니다.


## 디렉터리
```
src/orchestration/        오케스트레이션 패키지 (config·accompaniment·voice·mixing·pipeline)
run.py                CLI 진입점
app.py                (옵션) Gradio 통합 데모
src/          vendored: m2a_transformer/, tnp_voice_conversion/
configs/config.yaml   M2A 하이퍼파라미터
checkpoints/m2a/      M2A .ckpt  (gitignore — checkpoints/m2a/README.md 참고)
checkpoints/tnp/      TNP best.pt (gitignore — checkpoints/tnp/README.md 참고)
references/           목표 화자 reference WAV (gitignore — references/README.md 참고)
```

## 설치

```bash
bash scripts/setup_venv.sh          # uv 없으면 자동 설치, .venv 생성
source .venv/bin/activate           # Windows Git Bash: source .venv/Scripts/activate
```
스크립트 내부 순서 (수동 설치 시 동일):
```bash
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
uv pip install --only-binary=numpy -e .   # pyproject.toml 의존성 전체 + numpy wheel 강제
uv pip install --no-deps basic-pitch==0.4.0              # TF 없이 ONNX 백엔드
# 사운드폰트 없으면 스크립트가 fallback 다운로드
```
> CPU-only: `TORCH_INDEX=https://download.pytorch.org/whl/cpu bash scripts/setup_venv.sh`

### ⚠️ 단일 환경 의존성 충돌 (해결 완료 — 이렇게 풀었음)
한 env에 두 모델을 합칠 때 위험 2가지 + 보너스 1가지가 있었고, **end-to-end로 검증**해 해결했습니다:

| 위험 | 내용 | 해결책 (이 repo 적용·검증됨) |
|------|------|----------------------|
| 🔴 **A. basic-pitch ↔ TensorFlow** | basic-pitch 0.4.0이 py3.11+/Windows에서 **무조건 `tensorflow<2.15.1` 요구**(py3.12 wheel 없음, torch와도 충돌). `[onnx]` extra는 존재하지 않음 | **`pip install --no-deps basic-pitch==0.4.0` + `onnxruntime`** — wheel에 동봉된 `nmp.onnx`로 basic-pitch가 **ONNX 백엔드 자동 선택, TF 완전 배제** |
| 🟠 **B. deepfilternet 네이티브(Rust) 빌드** | `deepfilterlib`(Rust)는 Windows wheel이 **아예 없음** | **생략** — content encoder가 부재를 가드(`_denoise` passthrough), 학습도 skip_denoise → **deepfilternet 없이 음성변환 정상 동작 확인**. 노이즈 입력 denoise 필요 시 Linux/WSL에서만 설치 |
| 🟢 **C. transformers ↔ torch<2.6** | transformers 4.49+가 CVE로 `torch.load(.bin)`을 차단 → ContentVec(`.bin`) 로드 실패 | **`transformers==4.46.3` 핀** (가드 이전 버전). torch 업그레이드 불필요 |

추가로 numpy는 `--only-binary`로 wheel 강제(py3.12에서 낡은 numpy sdist 빌드가 setuptools를 깨뜨림),
TNP **FastAPI 서버는 미사용**(오프라인 `convert`만)이라 gradio↔fastapi 충돌도 원천 제거했습니다.

## 준비물
- `checkpoints/m2a/…ckpt` — M2A 모델 체크포인트
- (음성 변환 시) `checkpoints/tnp/latest.pt`(또는 `best.pt`) + 목표 화자 reference WAV
  — 없으면 **VC 건너뜀**(입력 보컬로 폴백)
- `soundfonts/default.sf2` — setup 스크립트가 받음(MIDI→WAV 렌더용)

> ⚠️ **reference 주의**: `--reference`에는 **변환 목표 화자**의 음성만 넣으세요
> (`target.wav` ✅, `context.wav` ✅). 원본 화자 `source.wav`나 모델 출력 `converted.wav`를
> 섞으면 화자 임베딩이 오염돼 변환이 망가집니다.

## 실행
```bash
# Branch A만 (TNP 준비 전에도 동작 — 입력 보컬 + 생성 반주 믹스)
python run.py --input references/source.wav \
    --m2a-checkpoint "checkpoints/m2a/best-epoch=007-val_loss=0.8431.ckpt" \
    --out output/run1

# 전체 (TNP 체크포인트 + 목표 화자 reference)
python run.py --input references/source.wav \
    --m2a-checkpoint "checkpoints/m2a/best-epoch=007-val_loss=0.8431.ckpt" \
    --tnp-checkpoint checkpoints/tnp/latest.pt \
    --reference references/target.wav \
    --out output/run2
```

### 출력 (`output/<run>/`)
| 파일 | 내용 |
|------|------|
| `01_input.wav` | 입력 사본 |
| `02_melody.mid` | 전사된 멜로디 MIDI |
| `03_accompaniment.mid/.wav` | 생성 반주 |
| `04_vocal.wav` | 변환 보컬 (VC 미적용 시 입력 폴백) |
| **`05_final_mix.wav`** | **최종 결과물 (보컬 + 반주)** |

## (옵션) Gradio 데모
```bash
python app.py            # 체크포인트 자동 고정 (CLI 인자 불필요)
```
체크포인트는 **자동 고정**됩니다 — `checkpoints/m2a/` 의 ckpt를 M2A로 시작 시 로드하고,
`checkpoints/tnp/` 의 ckpt를 TNP로 사용합니다(없으면 음성변환 건너뜀).
- **모드를 맨 위에서 선택**하면 입력/출력 컴포넌트가 모드에 맞게 바뀝니다 (UI 라디오 / CLI `--mode`):
  - **M2A + TNP (전체)** — 반주 + 음성 변환 → 변환보컬+반주 믹스 (출력: 멜로디·반주·보컬·믹스)
  - **M2A만** — 반주 생성 → 원본입력(또는 입력MIDI)+반주 믹스. **WAV 또는 MIDI 입력** 가능
    (MIDI면 전사 생략 → **MIDI→MIDI 검증**). 디버깅용 중간 산출물 노출.
  - **TNP만** — 음성 변환된 보컬만 (reference 필요)
- **중간 산출물(디버깅)**: `02_melody.wav`(전사/입력 MIDI를 멜로디 악기로 렌더 → 들어보기),
  `03_accompaniment.wav`(**순수 반주 — 멜로디 분리됨**), `05_final_mix.wav`, 멜로디/반주 **MIDI 다운로드**.
- **WAV→MIDI 전사기 선택** (`--transcriber`, UI 드롭다운):
  - **basic-pitch** — 다성 AMT(기본). 합주/다성 입력에 적합.
  - **crepe** — 단성 CREPE(f0→평활→노트 분할). **솔로 보컬의 비브라토/벤딩 분절을 크게 줄임**(예: 8→2노트).
- 변환 보컬은 후처리본(`04_vocal_post.wav`)을 재생/믹스에 사용하고, 원본 `04_vocal.wav`도 보존(A/B).
- 다른 ckpt를 쓰려면 `--m2a-checkpoint` / `--tnp-checkpoint` 로 직접 지정 가능.

> 즉 체크포인트 파일을 `checkpoints/m2a/`, `checkpoints/tnp/` 아래에 두기만 하면 됩니다.

> ⚠️ **파일명 주의**: 입력/레퍼런스 WAV 파일명에 `#`, `&`, `?`, `%`, `+` 같은 URL 특수문자가 있으면
> 브라우저 **미리듣기가 안 됩니다**(Gradio가 파일을 URL로 서빙 → URL이 깨짐). 처리·결과물은 정상이지만,
> 미리들으려면 파일명에서 이런 문자를 빼세요 (예: `A#_minor` → `Asharp_minor`).

### 속도 (모델 캐싱 + 시작 시 워밍업)
- **시작 시** M2A·TNP 모델을 모두 로드하고 basic-pitch 세션까지 워밍업합니다(콜드 비용을 부팅으로 이동).
- 단계별 소요 시간은 로그에 `▶ 시작 / ■ 완료 (Xs)` 로 출력됩니다.
- 모델 캐싱으로 **2회차 생성부터 ~2.7초**(콜드 ~32초 대비).

**워밍업 모드 — VRAM에 따라 자동 분기:**

| 모드 | 동작 | 첫 생성 | 적합 |
|------|------|---------|------|
| **light** (기본, VRAM<7GiB) | 모델 로드 + basic-pitch(CPU)만, GPU 더미 추론 X | ~7초 | dev (예: 4GB) — **OOM 방지** |
| **full** (VRAM≥7GiB 자동, 또는 `--full-warmup`) | 위 + 실제 1회 GPU 추론까지 워밍 | ~2초 | **시연용 큰 VRAM** |

- VRAM ≥ ~7GiB이면 **full 워밍업이 자동**으로 켜져 첫 생성도 빠릅니다. `--full-warmup`로 강제 ON,
  `--no-full-warmup`로 강제 OFF, `--no-warmup`로 전체 워밍업 OFF.
> 4GB 같은 작은 GPU에서 full을 강제하면 VRAM 피크가 넘쳐 OOM날 수 있어 기본은 light입니다
> (full 워밍업의 GPU 추론은 try/except로 감싸 실패해도 서버는 정상 기동).

## 동작 메모
- WAV 렌더링에는 시스템 `fluidsynth` + GM 사운드폰트가 필요합니다(없으면 MIDI만,
  믹스는 생략). M2A 원본 README의 사운드폰트 안내를 따르세요.
- TNP 인코더(ContentVec/crepe/deepfilternet/vocos)는 실용 속도에 **GPU 권장**.
