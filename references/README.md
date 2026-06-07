# 목표 화자 reference 음성 (자리표시자)

여기에 **변환 목표 화자**의 깨끗한 음성 샘플 WAV 를 1개 이상 두세요.

- 형식: WAV/FLAC (자동으로 16 kHz mono 로 리샘플됨)
- 여러 파일을 두면 화자 임베딩이 더 안정적입니다.
- 경로 예: `references/target1.wav`, `references/target2.wav`

이 파일이 없으면 음성 변환을 건너뜁니다(입력 보컬 그대로 사용).
실행 시 `--reference references/target1.wav references/target2.wav` 로 전달.
