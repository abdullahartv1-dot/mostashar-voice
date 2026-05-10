import { useState, useCallback } from 'react';
import { Download, Loader2, CheckCircle2 } from 'lucide-react';

interface TrimDownloaderProps {
  audioBuffer: AudioBuffer | null;
  selectionStart: number;
  selectionEnd: number;
  fileName: string;
  visible: boolean;
}

// Convert AudioBuffer to WAV format
function audioBufferToWav(buffer: AudioBuffer): ArrayBuffer {
  const numChannels = buffer.numberOfChannels;
  const sampleRate = buffer.sampleRate;
  const format = 1; // PCM
  const bitDepth = 16;
  const bytesPerSample = bitDepth / 8;

  // Interleave channels
  const channelData: Float32Array[] = [];
  for (let i = 0; i < numChannels; i++) {
    channelData.push(buffer.getChannelData(i));
  }

  const numSamples = channelData[0].length;
  const blockAlign = numChannels * bytesPerSample;
  const byteRate = sampleRate * blockAlign;
  const dataSize = numSamples * blockAlign;

  const headerSize = 44;
  const arrayBuffer = new ArrayBuffer(headerSize + dataSize);
  const view = new DataView(arrayBuffer);

  // Write WAV header
  const writeString = (offset: number, string: string) => {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  };

  writeString(0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, format, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitDepth, true);
  writeString(36, 'data');
  view.setUint32(40, dataSize, true);

  // Write interleaved audio data
  let offset = 44;
  for (let i = 0; i < numSamples; i++) {
    for (let ch = 0; ch < numChannels; ch++) {
      const sample = Math.max(-1, Math.min(1, channelData[ch][i]));
      const intSample = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
      view.setInt16(offset, intSample, true);
      offset += 2;
    }
  }

  return arrayBuffer;
}

export default function TrimDownloader({
  audioBuffer,
  selectionStart,
  selectionEnd,
  fileName,
  visible,
}: TrimDownloaderProps) {
  const [status, setStatus] = useState<'idle' | 'processing' | 'done'>('idle');

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = (seconds % 60).toFixed(2);
    return `${mins}:${secs.padStart(5, '0')}`;
  };

  const handleTrim = useCallback(async () => {
    if (!audioBuffer || status === 'processing') return;

    setStatus('processing');

    try {
      // Create offline audio context for rendering
      const offlineCtx = new OfflineAudioContext(
        audioBuffer.numberOfChannels,
        Math.ceil((selectionEnd - selectionStart) * audioBuffer.sampleRate),
        audioBuffer.sampleRate
      );

      const source = offlineCtx.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(offlineCtx.destination);
      source.start(0, selectionStart, selectionEnd - selectionStart);

      const renderedBuffer = await offlineCtx.startRendering();

      // Convert to WAV
      const wavData = audioBufferToWav(renderedBuffer);
      const blob = new Blob([wavData], { type: 'audio/wav' });
      const url = URL.createObjectURL(blob);

      // Trigger download
      const baseName = fileName.replace(/\.[^/.]+$/, '');
      const downloadName = `${baseName}_trimmed.wav`;

      const a = document.createElement('a');
      a.href = url;
      a.download = downloadName;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

      setStatus('done');
      setTimeout(() => setStatus('idle'), 3000);
    } catch (error) {
      console.error('Error trimming audio:', error);
      setStatus('idle');
      alert('Error processing audio. Please try again.');
    }
  }, [audioBuffer, selectionStart, selectionEnd, fileName, status]);

  if (!visible || !audioBuffer) return null;

  const duration = selectionEnd - selectionStart;

  return (
    <div className="mt-4 bg-[#1a1a1a] rounded-xl border border-[#2a2a2a] p-5 animate-fade-in-up">
      <p className="text-sm text-[#a0a0a0] mb-3">Selected Range:</p>
      <div className="flex items-center gap-3 font-mono text-sm text-[#f0f0f0] mb-4 flex-wrap">
        <span>Start: {formatTime(selectionStart)}</span>
        <span className="text-[#666666]">|</span>
        <span>End: {formatTime(selectionEnd)}</span>
        <span className="text-[#666666]">|</span>
        <span>Duration: {duration.toFixed(1)}s</span>
      </div>

      <button
        onClick={handleTrim}
        disabled={status === 'processing'}
        className={`w-full py-3 rounded-lg text-sm font-medium flex items-center justify-center gap-2 transition-all ${
          status === 'done'
            ? 'bg-[#22c55e] text-white'
            : 'bg-[#3b82f6] text-white hover:bg-[#2563eb] disabled:opacity-50'
        }`}
      >
        {status === 'processing' && <Loader2 className="w-4 h-4 spinner" />}
        {status === 'done' && <CheckCircle2 className="w-4 h-4" />}
        {status === 'processing' && 'Processing...'}
        {status === 'done' && 'Download started!'}
        {status === 'idle' && (
          <>
            <Download className="w-4 h-4" />
            Trim &amp; Download
          </>
        )}
      </button>
    </div>
  );
}
