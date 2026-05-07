"""VibeVoice ASR inference (no LoRA, just base model)"""
import argparse, json, sys, time
sys.stdout.reconfigure(encoding='utf-8')
import torch
from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_file', required=True)
    parser.add_argument('--base_model', default='microsoft/VibeVoice-ASR')
    parser.add_argument('--language_model', default='Qwen/Qwen2.5-7B')
    parser.add_argument('--max_new_tokens', type=int, default=8192)
    parser.add_argument('--context_info', default='Arabic interview, Saudi dialect')
    args = parser.parse_args()

    print(f'Loading processor and model (this may take 60-90s on first run)...')
    t0 = time.time()
    processor = VibeVoiceASRProcessor.from_pretrained(
        args.base_model,
        language_model_pretrained_name=args.language_model,
    )
    model = VibeVoiceASRForConditionalGeneration.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        attn_implementation='sdpa',
        trust_remote_code=True,
    ).to('cuda').eval()
    load_time = time.time() - t0
    print(f'  Loaded in {load_time:.1f}s')

    # Audio duration
    import subprocess, re, imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    r = subprocess.run([ffmpeg, '-i', args.audio_file], capture_output=True, text=True)
    m = re.search(r'Duration: (\d+):(\d+):([\d.]+)', r.stderr)
    duration = (int(m.group(1))*3600 + int(m.group(2))*60 + float(m.group(3))) if m else 0
    print(f'Audio: {args.audio_file} | Duration: {duration:.1f}s')

    print('Transcribing...')
    t0 = time.time()
    inputs = processor(
        audio_path=args.audio_file,
        return_tensors='pt',
        padding=True,
        add_generation_prompt=True,
        context_info=args.context_info,
    )
    inputs = {k: v.to('cuda') if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=processor.pad_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            do_sample=False,
        )

    input_length = inputs['input_ids'].shape[1]
    generated_ids = output_ids[0, input_length:]
    generated_text = processor.decode(generated_ids, skip_special_tokens=True)
    elapsed = time.time() - t0

    try:
        segments = processor.post_process_transcription(generated_text)
    except Exception as e:
        print(f'Parse error: {e}')
        segments = []

    print(f'\nDone in {elapsed:.1f}s ({duration/elapsed:.2f}x realtime)')
    print(f'Segments: {len(segments)}')
    print(f'\n=== Raw output (first 1500 chars) ===\n{generated_text[:1500]}')

    if segments:
        print(f'\n=== Structured output (first 30) ===')
        for s in segments[:30]:
            print(f"[{s.get('start_time', 'N/A')} - {s.get('end_time', 'N/A')}] "
                  f"Speaker {s.get('speaker_id', 'N/A')}: {s.get('text', '')[:100]}")

    # Save
    out = {
        'audio_file': args.audio_file,
        'duration': duration,
        'load_time': round(load_time, 2),
        'transcribe_time': round(elapsed, 2),
        'speedup': round(duration/elapsed, 2) if elapsed > 0 else 0,
        'raw_text': generated_text,
        'segments': segments,
    }
    out_path = args.audio_file.replace('.mp3', '_vibevoice_asr.json').replace('.mp4', '_vibevoice_asr.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
