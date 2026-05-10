import { useState, useCallback } from 'react';
import Sidebar from './components/Sidebar';
import UploadSection from './components/UploadSection';
import TipsCard from './components/TipsCard';
import WaveformEditor from './components/WaveformEditor';
import TrimDownloader from './components/TrimDownloader';

export default function App() {
  const [audioBuffer, setAudioBuffer] = useState<AudioBuffer | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectionStart, setSelectionStart] = useState(0);
  const [selectionEnd, setSelectionEnd] = useState(0);
  const [showTrimDownloader, setShowTrimDownloader] = useState(false);

  const handleFileSelect = useCallback((file: File, buffer: AudioBuffer) => {
    setSelectedFile(file);
    setAudioBuffer(buffer);

    // Default selection: first min(duration, 10) seconds, but at least 3 seconds
    const duration = buffer.duration;
    const defaultEnd = Math.min(duration, 10);
    const defaultStart = defaultEnd >= 3 ? 0 : Math.max(0, duration - 3);
    setSelectionStart(defaultStart);
    setSelectionEnd(Math.min(duration, defaultStart + 10));
    setShowTrimDownloader(false);
  }, []);

  const handleClearFile = useCallback(() => {
    setSelectedFile(null);
    setAudioBuffer(null);
    setSelectionStart(0);
    setSelectionEnd(0);
    setShowTrimDownloader(false);
  }, []);

  const handleSelectionChange = useCallback((start: number, end: number) => {
    setSelectionStart(start);
    setSelectionEnd(end);
  }, []);

  const handleTrim = useCallback(() => {
    setShowTrimDownloader(true);
  }, []);

  return (
    <div className="min-h-screen bg-[#0a0a0a]">
      {/* Sidebar */}
      <Sidebar />

      {/* Main Content */}
      <main className="ml-[240px] min-h-screen flex justify-center py-8 px-6">
        <div className="w-full max-w-[680px]">
          {/* Page Header */}
          <h1 className="text-[20px] font-semibold text-[#f0f0f0] mb-6 animate-fade-in-up">
            Instant Clone
          </h1>

          {/* Upload Section */}
          <UploadSection
            onFileSelect={handleFileSelect}
            selectedFile={selectedFile}
            onClearFile={handleClearFile}
          />

          {/* Tips Card */}
          <div className="mt-4">
            <TipsCard />
          </div>

          {/* Waveform Section */}
          <div className="mt-6 bg-[#1a1a1a] rounded-xl border border-[#2a2a2a] p-6">
            <WaveformEditor
              audioBuffer={audioBuffer}
              onSelectionChange={handleSelectionChange}
              onTrim={handleTrim}
              selectionStart={selectionStart}
              selectionEnd={selectionEnd}
            />
          </div>

          {/* Trim Downloader */}
          <TrimDownloader
            audioBuffer={audioBuffer}
            selectionStart={selectionStart}
            selectionEnd={selectionEnd}
            fileName={selectedFile?.name || 'audio.wav'}
            visible={showTrimDownloader}
          />
        </div>
      </main>
    </div>
  );
}
