import math

import librosa
import parselmouth
import soundfile as sf
from parselmouth.praat import call


def augment_pitch_and_formant(audio_path, output_path, formant_ratio, pitch_ratio):
    """
    formant_ratio: vocal-tract length scaling (e.g. 0.85 ~ 1.15)
    pitch_ratio: pitch scaling (e.g. 0.8 ~ 1.2 -> 1.2 raises pitch by 20%)
    """
    # 1. Load audio forced to mono
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    sound = parselmouth.Sound(y, sr)

    # 2. Extract the median pitch of the source audio
    pitch_obj = sound.to_pitch()
    median_pitch = call(pitch_obj, "Get quantile", 0.0, 0.0, 0.5, "Hertz")

    # 3. Guard: audio too short or silent so pitch can't be estimated (NaN)
    if math.isnan(median_pitch):
        target_pitch = 0.0  # skip pitch conversion
    else:
        target_pitch = median_pitch * pitch_ratio  # e.g. 150Hz * 1.2 = 180Hz

    # 4. Call Praat "Change Gender"
    # params: f0min, f0max, formant_ratio, target_pitch, pitch_range_ratio, duration_factor
    augmented_sound = call(
        sound, "Change gender", 75.0, 600.0, formant_ratio, target_pitch, 1.0, 1.0
    )

    # 5. Save safely with an integer sampling rate
    sf.write(
        output_path, augmented_sound.values[0], int(augmented_sound.sampling_frequency)
    )


# Quick run test
if __name__ == "__main__":
    # Raise formant 1.1x (light) and pitch 1.2x (higher) to build a target-speaker shield
    augment_pitch_and_formant(
        audio_path="datasets/wav48_silence_trimmed/p224/1_0000.wav",
        output_path="augmented_pitch_formant.wav",
        formant_ratio=0.7,
        pitch_ratio=0.9,
    )
