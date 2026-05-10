import { useRef, useEffect, useState, useCallback } from 'react';
import { Play, Pause } from 'lucide-react';

interface WaveformEditorProps {
  audioBuffer: AudioBuffer | null;
  onSelectionChange: (start: number, end: number) => void;
  onTrim: () => void;
  selectionStart: number;
  selectionEnd: number;
}

export default function WaveformEditor({
  audioBuffer,
  onSelectionChange,
  onTrim,
  selectionStart,
  selectionEnd,
}: WaveformEditorProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [waveformData, setWaveformData] = useState<number[]>([]);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1.0);
  const [dragState, setDragState] = useState<'none' | 'left' | 'right' | 'move'>('none');
  const [showTrimSection, setShowTrimSection] = useState(false);
  const audioSourceRef = useRef<AudioBufferSourceNode | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const startTimeRef = useRef<number>(0);
  const animationFrameRef = useRef<number>(0);
  const playheadRef = useRef<HTMLDivElement>(null);

  const duration = audioBuffer?.duration || 0;
  const selectionDuration = selectionEnd - selectionStart;
  const isOverLimit = selectionDuration > 10;
  const isUnderMin = selectionDuration < 3;

  // Generate waveform data
  useEffect(() => {
    if (!audioBuffer) return;

    const channelData = audioBuffer.getChannelData(0);
    const samples = 200;
    const blockSize = Math.floor(channelData.length / samples);
    const filteredData: number[] = [];

    for (let i = 0; i < samples; i++) {
      let sum = 0;
      for (let j = 0; j < blockSize; j++) {
        sum += Math.abs(channelData[i * blockSize + j]);
      }
      filteredData.push(sum / blockSize);
    }

    // Normalize
    const maxVal = Math.max(...filteredData, 0.01);
    const normalized = filteredData.map((v) => v / maxVal);
    setWaveformData(normalized);
  }, [audioBuffer]);

  // Draw waveform
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || waveformData.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;

    // Clear
    ctx.clearRect(0, 0, width, height);

    const barWidth = width / waveformData.length;
    const gap = 1;

    // Calculate selection pixel positions
    const selStartX = (selectionStart / duration) * width;
    const selEndX = (selectionEnd / duration) * width;

    // Draw unselected left region overlay
    if (selectionStart > 0) {
      ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
      ctx.fillRect(0, 0, selStartX, height);
    }

    // Draw unselected right region overlay
    if (selectionEnd < duration) {
      ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
      ctx.fillRect(selEndX, 0, width - selEndX, height);
    }

    // Draw selection highlight background
    ctx.fillStyle = 'rgba(59, 130, 246, 0.12)';
    ctx.fillRect(selStartX, 0, selEndX - selStartX, height);

    // Draw selection border
    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 2;
    ctx.strokeRect(selStartX, 0, selEndX - selStartX, height);

    // Draw waveform bars
    waveformData.forEach((value, index) => {
      const x = index * barWidth;
      const barHeight = value * height * 0.8;
      const y = (height - barHeight) / 2;

      const isInSelection = x >= selStartX && x <= selEndX;

      if (isInSelection) {
        ctx.fillStyle = '#f0f0f0';
      } else {
        ctx.fillStyle = '#555555';
      }

      ctx.fillRect(x + gap / 2, y, barWidth - gap, barHeight);
    });
  }, [waveformData, selectionStart, selectionEnd, duration]);

  // Handle mouse/touch events for dragging
  const getTimeFromX = useCallback((clientX: number) => {
    const canvas = canvasRef.current;
    if (!canvas || duration === 0) return 0;
    const rect = canvas.getBoundingClientRect();
    const x = clientX - rect.left;
    const ratio = Math.max(0, Math.min(1, x / rect.width));
    return ratio * duration;
  }, [duration]);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (!canvasRef.current || duration === 0) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const selStartX = (selectionStart / duration) * rect.width;
    const selEndX = (selectionEnd / duration) * rect.width;

    // Check if clicking on handles (10px tolerance)
    if (Math.abs(x - selStartX) <= 10) {
      setDragState('left');
    } else if (Math.abs(x - selEndX) <= 10) {
      setDragState('right');
    } else if (x > selStartX && x < selEndX) {
      setDragState('move');
    }
  }, [duration, selectionStart, selectionEnd]);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (dragState === 'none' || !audioBuffer) return;
    e.preventDefault();

    const time = getTimeFromX(e.clientX);
    let newStart = selectionStart;
    let newEnd = selectionEnd;

    if (dragState === 'left') {
      newStart = Math.max(0, Math.min(time, selectionEnd - 3));
      // Enforce max 10s limit
      if (selectionEnd - newStart > 10) {
        newStart = selectionEnd - 10;
      }
    } else if (dragState === 'right') {
      newEnd = Math.min(duration, Math.max(time, selectionStart + 3));
      // Enforce max 10s limit
      if (newEnd - selectionStart > 10) {
        newEnd = selectionStart + 10;
      }
    } else if (dragState === 'move') {
      const selDuration = selectionEnd - selectionStart;
      newStart = Math.max(0, time - selDuration / 2);
      newEnd = newStart + selDuration;
      if (newEnd > duration) {
        newEnd = duration;
        newStart = Math.max(0, newEnd - selDuration);
      }
    }

    onSelectionChange(newStart, newEnd);
    setShowTrimSection(false);
  }, [dragState, audioBuffer, getTimeFromX, selectionStart, selectionEnd, duration, onSelectionChange]);

  const handleMouseUp = useCallback(() => {
    setDragState('none');
  }, []);

  // Touch support
  const handleTouchStart = useCallback((e: React.TouchEvent) => {
    if (!canvasRef.current || duration === 0) return;
    const touch = e.touches[0];
    const rect = canvasRef.current.getBoundingClientRect();
    const x = touch.clientX - rect.left;
    const selStartX = (selectionStart / duration) * rect.width;
    const selEndX = (selectionEnd / duration) * rect.width;

    if (Math.abs(x - selStartX) <= 15) {
      setDragState('left');
    } else if (Math.abs(x - selEndX) <= 15) {
      setDragState('right');
    } else if (x > selStartX && x < selEndX) {
      setDragState('move');
    }
  }, [duration, selectionStart, selectionEnd]);

  const handleTouchMove = useCallback((e: React.TouchEvent) => {
    if (dragState === 'none' || !audioBuffer) return;
    e.preventDefault();
    const touch = e.touches[0];
    const time = getTimeFromX(touch.clientX);
    let newStart = selectionStart;
    let newEnd = selectionEnd;

    if (dragState === 'left') {
      newStart = Math.max(0, Math.min(time, selectionEnd - 3));
      if (selectionEnd - newStart > 10) {
        newStart = selectionEnd - 10;
      }
    } else if (dragState === 'right') {
      newEnd = Math.min(duration, Math.max(time, selectionStart + 3));
      if (newEnd - selectionStart > 10) {
        newEnd = selectionStart + 10;
      }
    }

    onSelectionChange(newStart, newEnd);
    setShowTrimSection(false);
  }, [dragState, audioBuffer, getTimeFromX, selectionStart, selectionEnd, duration, onSelectionChange]);

  const handleTouchEnd = useCallback(() => {
    setDragState('none');
  }, []);

  // Update cursor style
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    if (dragState === 'left' || dragState === 'right') {
      canvas.style.cursor = 'col-resize';
    } else if (dragState === 'move') {
      canvas.style.cursor = 'grabbing';
    } else {
      canvas.style.cursor = 'default';
    }
  }, [dragState]);

  // Playback
  const playSelection = useCallback(() => {
    if (!audioBuffer || isPlaying) {
      // Stop playback
      if (audioSourceRef.current) {
        audioSourceRef.current.stop();
        audioSourceRef.current = null;
      }
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      setIsPlaying(false);
      if (playheadRef.current) {
        playheadRef.current.style.display = 'none';
      }
      return;
    }

    const ctx = new AudioContext();
    audioContextRef.current = ctx;
    const source = ctx.createBufferSource();
    source.buffer = audioBuffer;
    source.playbackRate.value = playbackSpeed;
    source.connect(ctx.destination);
    source.start(0, selectionStart, selectionDuration);
    audioSourceRef.current = source;
    startTimeRef.current = ctx.currentTime;
    setIsPlaying(true);

    if (playheadRef.current) {
      playheadRef.current.style.display = 'block';
    }

    const updatePlayhead = () => {
      if (!audioContextRef.current || !playheadRef.current || !canvasRef.current) return;
      const elapsed = audioContextRef.current.currentTime - startTimeRef.current;
      const currentTime = selectionStart + elapsed;

      if (currentTime >= selectionEnd) {
        setIsPlaying(false);
        playheadRef.current.style.display = 'none';
        return;
      }

      const canvas = canvasRef.current;
      const rect = canvas.getBoundingClientRect();
      const x = ((currentTime / duration) * rect.width);
      playheadRef.current.style.left = `${x}px`;

      animationFrameRef.current = requestAnimationFrame(updatePlayhead);
    };

    animationFrameRef.current = requestAnimationFrame(updatePlayhead);

    source.onended = () => {
      setIsPlaying(false);
      if (playheadRef.current) {
        playheadRef.current.style.display = 'none';
      }
    };
  }, [audioBuffer, isPlaying, selectionStart, selectionDuration, playbackSpeed, duration]);

  // Format time display
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = (seconds % 60).toFixed(2);
    return `${mins}:${secs.padStart(5, '0')}`;
  };

  // Handle "Use X seconds" button
  const handleUseSelection = () => {
    if (isOverLimit || isUnderMin || !audioBuffer) return;
    setShowTrimSection(true);
    onTrim();
  };

  if (!audioBuffer) {
    return (
      <div className="text-center py-8 animate-fade-in-up" style={{ animationDelay: '200ms' }}>
        <p className="text-sm text-[#666666]">Upload an audio file to see the waveform</p>
      </div>
    );
  }

  return (
    <div className="animate-fade-in-up" style={{ animationDelay: '200ms' }}>
      {/* Validation Error Message */}
      {isOverLimit && (
        <div className="mb-3 animate-slide-down">
          <span className="inline-block bg-[rgba(239,68,68,0.1)] text-[#ef4444] text-xs px-3 py-1.5 rounded-md">
            Provide at most 10 seconds of audio
          </span>
        </div>
      )}
      {isUnderMin && (
        <div className="mb-3 animate-slide-down">
          <span className="inline-block bg-[rgba(239,68,68,0.1)] text-[#ef4444] text-xs px-3 py-1.5 rounded-md">
            Provide at least 3 seconds of audio
          </span>
        </div>
      )}

      {/* Section Label */}
      <p className="text-sm text-[#a0a0a0] text-center mb-4">
        Select 3-10 seconds of audio to use
      </p>

      {/* Waveform Container */}
      <div
        ref={containerRef}
        className="relative w-full h-[120px] mb-4 select-none"
        style={{ touchAction: 'none' }}
      >
        <canvas
          ref={canvasRef}
          className="w-full h-full rounded"
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          onTouchStart={handleTouchStart}
          onTouchMove={handleTouchMove}
          onTouchEnd={handleTouchEnd}
        />

        {/* Left Handle */}
        <div
          className="drag-handle drag-handle-left"
          style={{
            left: `${(selectionStart / duration) * 100}%`,
          }}
          onMouseDown={(e) => {
            e.stopPropagation();
            setDragState('left');
          }}
        />

        {/* Right Handle */}
        <div
          className="drag-handle drag-handle-right"
          style={{
            left: `${(selectionEnd / duration) * 100}%`,
          }}
          onMouseDown={(e) => {
            e.stopPropagation();
            setDragState('right');
          }}
        />

        {/* Playhead */}
        <div
          ref={playheadRef}
          className="playhead"
        />
      </div>

      {/* Controls Row */}
      <div className="flex items-center justify-between">
        {/* Length Display */}
        <div className="flex flex-col">
          <span className="text-xs text-[#a0a0a0]">Length</span>
          <span
            className={`font-mono text-sm font-medium ${
              isOverLimit || isUnderMin ? 'text-[#ef4444]' : 'text-[#f0f0f0]'
            }`}
          >
            {selectionDuration.toFixed(2)}s
          </span>
        </div>

        {/* Center: Speed Slider + Play */}
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <span className="text-xs text-[#666666]">Slow</span>
            <input
              type="range"
              min="0.5"
              max="2"
              step="0.1"
              value={playbackSpeed}
              onChange={(e) => setPlaybackSpeed(parseFloat(e.target.value))}
              className="speed-slider w-24"
            />
            <span className="text-xs text-[#666666]">Fast</span>
          </div>
          <span className="text-xs text-[#a0a0a0] w-10 text-center">
            {playbackSpeed.toFixed(2)}x
          </span>

          {/* Play Button */}
          <button
            onClick={playSelection}
            className="w-9 h-9 rounded-full bg-white flex items-center justify-center hover:bg-[#f0f0f0] transition-colors shadow-md"
          >
            {isPlaying ? (
              <Pause className="w-4 h-4 text-black" />
            ) : (
              <Play className="w-4 h-4 text-black ml-0.5" />
            )}
          </button>
        </div>

        {/* Use X Seconds Button */}
        <button
          onClick={handleUseSelection}
          disabled={isOverLimit || isUnderMin}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
            isOverLimit || isUnderMin
              ? 'bg-[#3b82f6] text-white opacity-50 cursor-not-allowed'
              : 'bg-[#3b82f6] text-white hover:bg-[#2563eb]'
          }`}
        >
          {isOverLimit
            ? 'Max 10 seconds'
            : isUnderMin
            ? 'Min 3 seconds'
            : `Use ${Math.round(selectionDuration)} seconds`}
        </button>
      </div>

      {/* Trim Confirmation Section */}
      {showTrimSection && (
        <div className="mt-4 bg-[#1a1a1a] rounded-xl border border-[#2a2a2a] p-5 animate-fade-in-up">
          <p className="text-sm text-[#a0a0a0] mb-3">Selected Range:</p>
          <div className="flex items-center gap-4 font-mono text-sm text-[#f0f0f0] mb-4">
            <span>Start: {formatTime(selectionStart)}</span>
            <span className="text-[#666666]">|</span>
            <span>End: {formatTime(selectionEnd)}</span>
            <span className="text-[#666666]">|</span>
            <span>Duration: {selectionDuration.toFixed(1)}s</span>
          </div>
          <button
            onClick={onTrim}
            className="w-full py-3 bg-[#3b82f6] text-white rounded-lg text-sm font-medium hover:bg-[#2563eb] transition-colors"
          >
            Trim &amp; Download
          </button>
        </div>
      )}
    </div>
  );
}
