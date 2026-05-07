import sys
import time
import shutil
from pathlib import Path
from gradio_client import Client, handle_file

sys.stdout.reconfigure(encoding='utf-8')

OUT_DIR = Path('C:/Users/Admin/Desktop/project/x')

# Speaker 1 = MUJIB (الضيف الذي قال "كنت مليونير...")
# Speaker 2 = MUHAWIR (المُحاوِر الذي يسأل)
sp1_audio = str(OUT_DIR / 'SPEAKER_1_sample.wav')  # 13.74s
sp2_audio = str(OUT_DIR / 'SPEAKER_0_sample.wav')  # 7.44s

# Use both originals to compare:
# - For Speaker 1 (الضيف), generate text similar to what they said
# - For Speaker 2 (المحاور), generate a question
text = """[1]: مرحبا، أنا أحدثك بصوتي الأصلي. لقد كنت ناجحا في الاستثمار وحققت أرباحا كبيرة في وقت قصير.
[2]: ممتاز جدا، كم نسبة الربح التي حققتها؟ وما هو السر؟"""

print('Connecting to VibeVoice API...')
client = Client('vibingvoice/vibe-voice-custom-voices')

print('Generating cloned speech (this may take 30-60s on ZeroGPU)...')
t0 = time.time()
result = client.predict(
    text=text,
    speaker1_audio_path=handle_file(sp1_audio),
    speaker2_audio_path=handle_file(sp2_audio),
    speaker3_audio_path=handle_file(sp1_audio),  # placeholder
    speaker4_audio_path=handle_file(sp2_audio),  # placeholder
    seed=42,
    diffusion_steps=20,
    cfg_scale=1.3,
    use_sampling=False,
    temperature=0.95,
    top_p=0.95,
    max_words_per_chunk=250,
    api_name='/generate_speech_gradio',
)
elapsed = time.time() - t0
print(f'Done in {elapsed:.1f}s')
print(f'Result file: {result}')

# Copy to our project dir
out_path = OUT_DIR / 'cloned_output.wav'
shutil.copy(result, out_path)
print(f'Saved: {out_path}')
print(f'Size: {out_path.stat().st_size / 1024:.1f} KB')
