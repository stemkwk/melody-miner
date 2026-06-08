# Melody Miner - 분석 가이드 (Analysis Guide)

이 폴더(`analysis/`)는 모델의 결과물을 분석하고 시각화하기 위한 스크립트의 결과물과 가이드를 보관하는 곳입니다. 파이프라인의 두 가지 주요 브랜치(Branch A: 반주 생성, Branch B: 음성 변환)를 시각적으로 검증할 수 있는 도구들이 준비되어 있습니다.

## 1. 피아노 롤 시각화 (반주 생성 분석)

입력 멜로디와 모델이 생성한 반주(Accompaniment)를 피아노 롤(Piano Roll) 형태로 겹쳐서 시각화합니다. 발표 자료에서 단선율 멜로디 주변에 화음과 베이스가 어떻게 생성되어 깔리는지 직관적으로 보여줄 때 가장 적합합니다.

- **스크립트**: `scripts/visualize_piano_roll.py`
- **기본 실행 (최신 결과물 자동 분석)**
  ```bash
  python scripts/visualize_piano_roll.py
  ```
  별도의 인자를 주지 않으면 `output/` 폴더 내에 있는 가장 최근 작업물(`03_accompaniment_full.mid`)을 자동으로 찾아서 `analysis/piano_roll/piano_roll.png` 파일로 시각화합니다.
- **특정 파일 직접 분석**
  원하는 과거의 미디 파일이 있다면 `--midi` 파라미터로 지정할 수 있습니다.
  ```bash
  python scripts/visualize_piano_roll.py --midi "output/특정폴더/03_accompaniment_full.mid"
  ```
- **분석 산출물 (`analysis/piano_roll/piano_roll.png`)**: 
  - **빨간색 선**: 원본 입력 멜로디 트랙 (Input Melody)
  - **파란색 선**: 모델이 생성한 반주 트랙 (Generated Accompaniment)

---

## 2. 보컬 음성 변환 진단 (스펙트로그램 분석)

음성 변환 모델(TNP)이 보컬을 변환할 때 스펙트로그램의 디테일이 뭉개지거나(Smoothing) 과하게 떨리는(Wobble) 현상을 진단합니다.

- **스크립트**: `scripts/diagnose_vocoder_wobble.py`
- **기본 실행 (기본 레퍼런스 분석)**
  ```bash
  python scripts/diagnose_vocoder_wobble.py
  ```
  기본값으로 `references/source.wav` (입력 보컬)와 `references/context.wav` (타겟 보컬)를 사용하여 진단을 수행합니다.
- **특정 파일 직접 분석**
  실제 변환에 사용하려는 입력 보컬과 목표 화자의 오디오 파일을 명시하여 테스트할 수 있습니다.
  ```bash
  python scripts/diagnose_vocoder_wobble.py --source "경로/다른_입력보컬.wav" --reference "경로/다른_목표화자.wav"
  ```
- **분석 산출물 (`analysis/diag/` 폴더 내부)**:
  - `mel_real_input.png`, `mel_pred_convert.png`: 변환 전후의 스펙트로그램 시각화 이미지
  - `mel_compare.png`: 두 스펙트로그램의 대조 이미지 및 평활화 정도를 수치(비율)로 보여주는 리포트
  - `convert_sharp_*.wav`: Mel Sharpening 강도를 조절해가며 생성한 오디오 파일. 이 파일들을 직접 들어보고 가장 자연스러운 오디오의 설정값(`t_alpha`, `f_alpha`)을 찾은 뒤, `src/orchestration/voice.py`에 반영하면 실제 파이프라인의 음질을 개선할 수 있습니다.
