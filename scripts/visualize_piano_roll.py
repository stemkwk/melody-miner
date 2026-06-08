import argparse
from pathlib import Path
import pretty_midi
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def plot_piano_roll(midi_data, out_path, title="Melody vs Accompaniment Piano Roll"):
    plt.figure(figsize=(16, 7))
    
    # 멜로디(입력)와 반주(생성)를 시각적으로 구분하기 위한 색상 설정
    melody_color = '#E63946'  # 강렬한 빨간색 (멜로디 강조)
    accomp_color = '#457B9D'  # 차분한 파란색 (반주 배경)
    
    max_time = midi_data.get_end_time()
    min_pitch = 127
    max_pitch = 0
    
    # 각 악기 트랙별로 노트 그리기
    for i, instrument in enumerate(midi_data.instruments):
        track_name = instrument.name.upper() if instrument.name else f"TRACK {i}"
        
        if "MELODY" in track_name:
            color = melody_color
            zorder = 10
            alpha = 1.0
            label = "Input Melody"
        else:
            color = accomp_color
            zorder = 5
            alpha = 0.8
            label = "Generated Accompaniment"
            
        for note in instrument.notes:
            min_pitch = min(min_pitch, note.pitch)
            max_pitch = max(max_pitch, note.pitch)
            # 노트의 시작~끝을 선으로 그림
            plt.plot(
                [note.start, note.end],
                [note.pitch, note.pitch],
                color=color,
                linewidth=6,
                alpha=alpha,
                solid_capstyle='round',
                zorder=zorder
            )
            
        # 범례를 위해 보이지 않는 선 추가
        if len(instrument.notes) > 0:
            plt.plot([], [], color=color, linewidth=6, label=label)
            
    # Y축(음높이) 범위를 실제 연주된 범위에 맞게 조절 (위아래 여백 4 추가)
    plt.ylim(max(0, min_pitch - 4), min(127, max_pitch + 4))
    plt.xlim(0, max_time)
    
    # 배경색 및 그리드 스타일링
    plt.gca().set_facecolor('#F8F9FA')
    plt.grid(True, axis='both', linestyle='--', alpha=0.6, color='#CED4DA')
    
    plt.xlabel("Time (seconds)", fontsize=14, fontweight='bold', color='#495057')
    plt.ylabel("Pitch (MIDI Note)", fontsize=14, fontweight='bold', color='#495057')
    plt.title(title, fontsize=18, fontweight='bold', pad=20, color='#212529')
    
    # 중복 라벨 제거 후 범례 표시
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=12, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Piano roll saved to {out_path}")

def get_latest_midi():
    # output 폴더 내의 가장 최근 03_accompaniment_full.mid 찾기
    out_dir = Path("output")
    if not out_dir.exists(): return ""
    mids = list(out_dir.glob("*/03_accompaniment_full.mid"))
    if not mids: return "03_accompaniment_full.mid"
    return str(max(mids, key=lambda p: p.stat().st_mtime))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Piano Roll visualization from MIDI")
    parser.add_argument("--midi", default=get_latest_midi(), help="Path to the combined MIDI file (e.g., 03_accompaniment_full.mid)")
    parser.add_argument("--out", default="analysis/piano_roll/piano_roll.png", help="Output image file path")
    args = parser.parse_args()
    
    if not args.midi:
        print("Error: No midi file provided and could not find any in output/")
        exit(1)
        
    midi_data = pretty_midi.PrettyMIDI(args.midi)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_piano_roll(midi_data, args.out)
