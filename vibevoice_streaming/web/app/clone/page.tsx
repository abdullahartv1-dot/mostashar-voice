"use client"

import * as React from "react"
import { useRouter } from "next/navigation"
import { Upload, Mic, Square, Play, Pause } from "lucide-react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

export default function ClonePage() {
  const router = useRouter()
  const [mode, setMode] = React.useState<"upload" | "record">("upload")
  const [file, setFile] = React.useState<File | null>(null)
  const [audioBuffer, setAudioBuffer] = React.useState<AudioBuffer | null>(null)
  const [region, setRegion] = React.useState({ start: 0, end: 10 })
  const [name, setName] = React.useState("")
  const [language, setLanguage] = React.useState("ar")
  const [description, setDescription] = React.useState("")
  const [recording, setRecording] = React.useState(false)
  const [busy, setBusy] = React.useState(false)
  const [error, setError] = React.useState<string | null>(null)
  const recRef = React.useRef<MediaRecorder | null>(null)
  const recChunksRef = React.useRef<Blob[]>([])

  async function handleFile(f: File) {
    setFile(f); setError(null)
    const arr = await f.arrayBuffer()
    const ctx = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 })
    const buf = await ctx.decodeAudioData(arr.slice(0))
    setAudioBuffer(buf)
    setRegion({ start: 0, end: Math.min(10, buf.duration) })
  }

  async function toggleRecord() {
    if (!recording) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
        const r = new MediaRecorder(stream)
        recChunksRef.current = []
        r.ondataavailable = (e) => recChunksRef.current.push(e.data)
        r.onstop = async () => {
          stream.getTracks().forEach(t => t.stop())
          const blob = new Blob(recChunksRef.current, { type: "audio/webm" })
          await handleFile(new File([blob], "recording.webm", { type: "audio/webm" }))
        }
        recRef.current = r; r.start(); setRecording(true)
      } catch (e: any) { setError("فشل المايك: " + e.message) }
    } else {
      recRef.current?.stop(); setRecording(false)
    }
  }

  async function handleClone() {
    if (!file) return setError("ارفع/سجّل مقطع صوتي")
    if (!name.trim()) return setError("أدخل اسم الصوت")
    const dur = region.end - region.start
    if (dur < 3 || dur > 30) return setError("اختر بين 3 و 30 ثانية")
    setBusy(true); setError(null)
    const fd = new FormData()
    fd.append("name", name); fd.append("language", language)
    fd.append("start_s", String(region.start)); fd.append("end_s", String(region.end))
    fd.append("audio", file)
    try {
      const r = await fetch("/voices/clone", { method: "POST", body: fd })
      const j = await r.json()
      if (!r.ok) throw new Error(j.detail || "فشل")
      router.push(`/library?selected=${j.voice_id}`)
    } catch (e: any) { setError("خطأ: " + e.message) } finally { setBusy(false) }
  }

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Instant Clone</h1>

      <div className="bg-card border border-border rounded-xl p-6 space-y-4">
        <label className="text-sm text-muted-foreground">المقطع <span className="text-destructive">*</span></label>

        <div className="grid grid-cols-2 gap-2">
          <button onClick={() => setMode("upload")} className={cn(
            "py-3 px-4 rounded-lg border text-sm font-medium transition-colors",
            mode === "upload" ? "border-primary bg-primary/10 text-foreground" : "border-border bg-secondary text-muted-foreground"
          )}>
            <Upload className="inline-block mr-2 size-4" />رفع ملف
          </button>
          <button onClick={() => setMode("record")} className={cn(
            "py-3 px-4 rounded-lg border text-sm font-medium transition-colors",
            mode === "record" ? "border-primary bg-primary/10 text-foreground" : "border-border bg-secondary text-muted-foreground"
          )}>
            <Mic className="inline-block mr-2 size-4" />تسجيل
          </button>
        </div>

        {mode === "upload" ? (
          <label className="block border-2 border-dashed border-border rounded-lg p-8 text-center cursor-pointer hover:border-primary hover:bg-primary/5 transition-colors">
            <input type="file" accept="audio/*" className="hidden" onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])} />
            <div className="font-semibold">اضغط لرفع مقطع صوتي</div>
            <div className="text-xs text-muted-foreground mt-1">MP3 / WAV / M4A — حد أقصى 4 MB</div>
            {file && <div className="mt-3 text-sm text-foreground">📁 {file.name}</div>}
          </label>
        ) : (
          <div className="text-center py-8">
            <Button size="lg" onClick={toggleRecord} variant={recording ? "destructive" : "default"}>
              {recording ? <><Square className="mr-2 size-4" />إيقاف التسجيل</> : <><Mic className="mr-2 size-4" />ابدأ التسجيل</>}
            </Button>
            {file && !recording && <div className="mt-3 text-sm text-muted-foreground">سُجّل: {(file.size/1024).toFixed(0)} KB</div>}
          </div>
        )}

        <div className="bg-secondary rounded-lg p-4 text-sm text-muted-foreground">
          <div className="font-semibold text-foreground mb-2">للحصول على أفضل نتيجة:</div>
          <ul className="space-y-1 list-none">
            <li>✓ تجنب الصمت الطويل والوقفات</li>
            <li>✓ استخدم وتيرة الصوت الذي تريد أن يطابقها الصوت المستنسخ</li>
            <li>✓ قص المقطع لإزالة الضوضاء قبل الإرسال</li>
          </ul>
        </div>

        {audioBuffer && (
          <Waveform audioBuffer={audioBuffer} region={region} onRegionChange={setRegion} />
        )}
      </div>

      <div className="bg-card border border-border rounded-xl p-6 space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="text-sm text-muted-foreground">الاسم <span className="text-destructive">*</span></label>
            <input
              type="text" value={name} onChange={(e) => setName(e.target.value)}
              placeholder="مثال: صوت أحمد"
              className="mt-1 w-full px-3 py-2 rounded-md border border-border bg-secondary text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring"
            />
          </div>
          <div>
            <label className="text-sm text-muted-foreground">اللغة <span className="text-destructive">*</span></label>
            <select
              value={language} onChange={(e) => setLanguage(e.target.value)}
              className="mt-1 w-full px-3 py-2 rounded-md border border-border bg-secondary text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
            >
              <option value="ar">العربية (Arabic)</option>
              <option value="en">English</option>
              <option value="multi">متعدد اللغات</option>
            </select>
          </div>
        </div>
        <div>
          <label className="text-sm text-muted-foreground">الوصف (اختياري)</label>
          <input
            type="text" value={description} onChange={(e) => setDescription(e.target.value)}
            placeholder="ملاحظات عن الصوت"
            className="mt-1 w-full px-3 py-2 rounded-md border border-border bg-secondary text-foreground"
          />
        </div>
        {error && <div className="text-sm text-destructive">{error}</div>}
        <div className="flex items-center gap-3 pt-2">
          <Button onClick={handleClone} disabled={busy} size="lg">
            {busy ? "جاري الاستنساخ..." : "🎯 استنساخ الصوت"}
          </Button>
          <span className="text-xs font-mono text-muted-foreground">POST /voices/clone</span>
        </div>
      </div>
    </div>
  )
}

function Waveform({ audioBuffer, region, onRegionChange }: {
  audioBuffer: AudioBuffer
  region: { start: number; end: number }
  onRegionChange: (r: { start: number; end: number }) => void
}) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null)
  const draw = React.useCallback(() => {
    const c = canvasRef.current; if (!c) return
    const dpr = window.devicePixelRatio || 1
    c.width = c.clientWidth * dpr; c.height = c.clientHeight * dpr
    const ctx = c.getContext("2d")!; ctx.scale(dpr, dpr)
    const w = c.clientWidth, h = c.clientHeight
    const data = audioBuffer.getChannelData(0)
    const samplesPerPx = Math.ceil(data.length / w)
    const dur = audioBuffer.duration
    ctx.fillStyle = "hsl(240 5% 10%)"; ctx.fillRect(0, 0, w, h)
    ctx.fillStyle = "hsl(240 4% 35%)"
    for (let x = 0; x < w; x++) {
      let max = 0
      for (let i = 0; i < samplesPerPx; i++) {
        const v = Math.abs(data[x * samplesPerPx + i] || 0); if (v > max) max = v
      }
      const bh = max * h * 0.85
      ctx.fillRect(x, h / 2 - bh / 2, 1, bh)
    }
    const x0 = (region.start / dur) * w, x1 = (region.end / dur) * w
    ctx.fillStyle = "hsla(217 95% 65% / 0.15)"
    ctx.fillRect(x0, 0, x1 - x0, h)
    ctx.strokeStyle = "hsl(217 95% 65%)"; ctx.lineWidth = 2
    ctx.strokeRect(x0, 0, x1 - x0, h)
    ctx.fillStyle = "hsl(217 95% 65%)"
    for (let x = x0; x < x1; x++) {
      let max = 0
      for (let i = 0; i < samplesPerPx; i++) {
        const v = Math.abs(data[Math.floor(x) * samplesPerPx + i] || 0); if (v > max) max = v
      }
      const bh = max * h * 0.85
      ctx.fillRect(x, h / 2 - bh / 2, 1, bh)
    }
  }, [audioBuffer, region])

  React.useEffect(() => { draw() }, [draw])
  React.useEffect(() => {
    const onResize = () => draw()
    window.addEventListener("resize", onResize)
    return () => window.removeEventListener("resize", onResize)
  }, [draw])

  function onMouseDown(e: React.MouseEvent) {
    const c = canvasRef.current; if (!c) return
    const rect = c.getBoundingClientRect()
    const startPx = e.clientX - rect.left
    const dur = audioBuffer.duration
    const startTime = (startPx / rect.width) * dur
    const onMove = (ev: MouseEvent) => {
      const movePx = ev.clientX - rect.left
      const moveTime = Math.max(0, Math.min(dur, (movePx / rect.width) * dur))
      let s = Math.min(startTime, moveTime), eend = Math.max(startTime, moveTime)
      if (eend - s > 30) eend = s + 30
      if (eend - s < 0.5) eend = Math.min(s + 0.5, dur)
      onRegionChange({ start: s, end: eend })
    }
    const onUp = () => {
      window.removeEventListener("mousemove", onMove)
      window.removeEventListener("mouseup", onUp)
    }
    window.addEventListener("mousemove", onMove)
    window.addEventListener("mouseup", onUp)
  }

  const dur = region.end - region.start
  const ok = dur >= 3 && dur <= 30
  return (
    <div className="bg-secondary rounded-lg p-4 space-y-2">
      <div className="text-xs text-muted-foreground">اختر 3 - 30 ثانية للاستنساخ (اسحب على الموجة)</div>
      <canvas ref={canvasRef} onMouseDown={onMouseDown} className="w-full h-24 rounded cursor-crosshair" />
      <div className="text-sm font-mono">
        {region.start.toFixed(1)}s → {region.end.toFixed(1)}s ({dur.toFixed(1)}s)
        {!ok && <span className="text-destructive ml-2">⚠ خارج النطاق المسموح</span>}
      </div>
    </div>
  )
}
