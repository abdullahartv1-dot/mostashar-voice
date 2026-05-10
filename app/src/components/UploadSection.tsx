import { useRef, useState, useCallback } from 'react';
import { Upload, Mic, FileAudio, X } from 'lucide-react';

interface UploadSectionProps {
  onFileSelect: (file: File, audioBuffer: AudioBuffer) => void;
  selectedFile: File | null;
  onClearFile: () => void;
}

export default function UploadSection({ onFileSelect, selectedFile, onClearFile }: UploadSectionProps) {
  const [activeTab, setActiveTab] = useState<'record' | 'upload'>('upload');
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const processFile = useCallback(async (file: File) => {
    if (!file.type.startsWith('audio/')) {
      alert('Please select an audio file');
      return;
    }
    if (file.size > 4 * 1024 * 1024) {
      alert('File size must be less than 4 MB');
      return;
    }

    try {
      const arrayBuffer = await file.arrayBuffer();
      const audioContext = new AudioContext();
      const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
      onFileSelect(file, audioBuffer);
    } catch (error) {
      console.error('Error decoding audio:', error);
      alert('Could not decode audio. Please try a different file.');
    }
  }, [onFileSelect]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) processFile(file);
  }, [processFile]);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) processFile(file);
    e.target.value = '';
  }, [processFile]);

  return (
    <div className="bg-[#1a1a1a] rounded-xl border border-[#2a2a2a] p-6 animate-fade-in-up">
      {/* Section Label */}
      <div className="flex items-center gap-2 mb-4">
        <span className="text-red-500 text-xs">*</span>
        <span className="text-sm font-medium text-[#f0f0f0]">clip</span>
      </div>

      {/* Tabs */}
      <div className="flex bg-[#222222] rounded-lg p-1 mb-4">
        <button
          onClick={() => setActiveTab('record')}
          className={`flex-1 flex items-center justify-center gap-2 py-2.5 rounded-md text-sm transition-all tab-button ${
            activeTab === 'record'
              ? 'bg-[#1a1a1a] text-[#f0f0f0] shadow-sm'
              : 'text-[#666666] hover:text-[#a0a0a0]'
          }`}
        >
          <Mic className="w-4 h-4" />
          Record
        </button>
        <button
          onClick={() => setActiveTab('upload')}
          className={`flex-1 flex items-center justify-center gap-2 py-2.5 rounded-md text-sm transition-all tab-button ${
            activeTab === 'upload'
              ? 'bg-[#1a1a1a] text-[#f0f0f0] shadow-sm'
              : 'text-[#666666] hover:text-[#a0a0a0]'
          }`}
        >
          <Upload className="w-4 h-4" />
          Upload
        </button>
      </div>

      {activeTab === 'upload' && (
        <div className="animate-fade-in">
          {/* Upload Label */}
          <p className="text-sm font-medium text-[#f0f0f0] mb-3">Upload an audio clip</p>

          {/* File Name Badge */}
          {selectedFile && (
            <div className="flex items-center gap-3 bg-[#222222] rounded-md px-3 py-2.5 mb-3 animate-fade-in">
              <FileAudio className="w-4 h-4 text-[#a0a0a0] flex-shrink-0" />
              <span className="text-sm font-mono text-[#f0f0f0] flex-1 truncate">
                {selectedFile.name}
              </span>
              <button
                onClick={onClearFile}
                className="text-[#666666] hover:text-[#ef4444] transition-colors p-0.5"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          )}

          {/* Drop Zone */}
          {!selectedFile && (
            <div
              onClick={() => fileInputRef.current?.click()}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              className={`upload-zone border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-all ${
                isDragOver
                  ? 'border-[#3b82f6] bg-[rgba(59,130,246,0.05)]'
                  : 'border-[#2a2a2a] hover:border-[#3b82f6] hover:bg-[rgba(59,130,246,0.02)]'
              }`}
            >
              <Upload className="w-8 h-8 text-[#a0a0a0] mx-auto mb-2" />
              <p className="text-sm text-[#a0a0a0]">
                Click to browse or drag and drop
              </p>
              <input
                ref={fileInputRef}
                type="file"
                accept="audio/*"
                onChange={handleFileInput}
                className="hidden"
              />
            </div>
          )}

          {/* File Size Note */}
          <p className="text-xs text-[#666666] mt-3">
            The maximum file size is 4 MB.
          </p>
        </div>
      )}

      {activeTab === 'record' && (
        <div className="text-center py-8 animate-fade-in">
          <Mic className="w-10 h-10 text-[#666666] mx-auto mb-3" />
          <p className="text-sm text-[#a0a0a0]">Recording feature coming soon</p>
        </div>
      )}
    </div>
  );
}
