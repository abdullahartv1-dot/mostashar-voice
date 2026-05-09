"use client"

import * as React from "react"
import { useSearchParams } from "next/navigation"
import { Trash2, Volume2, VolumeX } from "lucide-react"
import { Button } from "@/components/ui/button"
import { VoicePicker } from "@/components/ui/voice-picker"
import { Orb } from "@/components/ui/orb"
import { type Voice, type BackendVoice, backendToVoice } from "@/lib/types"
import { cn } from "@/lib/utils"

export default function LibraryPage() {
  const params = useSearchParams()
  const preselected = params.get("selected") || ""
  const [voices, setVoices] = React.useState<Voice[]>([])
  const [selected, setSelected] = React.useState<string>(preselected)
  const [search, setSearch] = React.useState("")
  const [text, setText] = React.useState("مرحبا، أنا صوتك المستنسخ. أستطيع قراءة أي نص بنفس بصمتك الصوتية.")
  const [pickerOpen, setPickerOpen] = React.useState(false)
  const [ttfa, setTtfa] = React.useState<number | null>(null)
  const [status, setStatus] = React.useState("جاهز")
  const [isPlaying, setIsPlaying] = React.useState(false)
  const wsRef = React.useRef<WebSocket | null>(null)
  const audioCtxRef = React.useRef<AudioContext | null>(null)
  const nextStartRef = React.useRef(0)
  const previewAudioRef = React.useRef<HTMLAudioElement | null>(null)

  const load = React.useCallback(async () => {
    const r = await fetch("/voices")
    const j = await r.json()
    const list: Voice[] = (j.voices as BackendVoice[]).map(backendToVoice)
    setVoices(list)
    if (!selected && list.length) setSelected(preselected || list[0].voiceId)
  }, [selected, preselected])

  React.useEffect(() => { load() }, [load])

  const filtered = voices.filter(v => v.name.toLowerCase().includes(search.toLowerCase()))
  const selectedVoice = voices.find(v => v.voiceId === selected)

  function previewVoice(v: Voice) {
    if (previewAudioRef.current) { previewAudioRef.current.pause(); previewAudioRef.current = null }
    if (!v.previewUrl) return
    const a = new Audio(v.previewUrl)
    previewAudioRef.current = a
    a.play()
  }

  async function deleteVoice(v: Voice) {
    if (v.voiceId === "default") return
    if (!confirm(`حذف "${v.name}"؟`)) return
    await fetch(`/voices/${v.voiceId}`, { method: "DELETE" })
    await load()
  }

  function pcm16ToFloat32(buf: ArrayBuffer) {
    const view = new DataView(buf)
    const len = buf.byteLength / 2
    const f32 = new Float32Array(len)
    for (let i = 0; i < len; i++) {
      const s = view.getInt16(i * 2, true)
      f32[i] = s < 0 ? s / 32768 : s / 32767
    }
    return f32
  }

  async function play() {
    if (!selectedVoice) return
    const t = text.trim(); if (!t) return
    setIsPlaying(true); setStatus("جارٍ التوليد..."); setTtfa(null)
    audioCtxRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 })
    nextStartRef.current = 0
    const url = (location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/tts/stream"
    const ws = new WebSocket(url)
    ws.binaryType = "arraybuffer"
    wsRef.current = ws
    const start = performance.now()
    ws.onopen = () => {
      ws.send(JSON.stringify({ text: t, voice_id: selectedVoice.voiceId }))
    }
    ws.onmessage = (ev) => {
      if (typeof ev.data === "string") {
        const m = JSON.parse(ev.data)
        if (m.type === "ttfa") { setTtfa(m.ms); setStatus("يُشغّل...") }
        else if (m.type === "done") { setStatus(`اكتمل: ${m.total_audio_s}s صوت في ${m.total_wall_ms}ms`) }
        else if (m.type === "error") { setStatus("خطأ: " + m.message); setIsPlaying(false) }
      } else {
        const f32 = pcm16ToFloat32(ev.data)
        const ctx = audioCtxRef.current!
        if (nextStartRef.current < ctx.currentTime) nextStartRef.current = ctx.currentTime + 0.05
        const buf = ctx.createBuffer(1, f32.length, 24000)
        buf.copyToChannel(f32, 0)
        const src = ctx.createBufferSource()
        src.buffer = buf; src.connect(ctx.destination); src.start(nextStartRef.current)
        nextStartRef.current += buf.duration
      }
    }
    ws.onclose = () => setIsPlaying(false)
  }

  function stop() {
    wsRef.current?.close()
    audioCtxRef.current?.close()
    setIsPlaying(false); setStatus("موقوف")
  }

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">مكتبة الأصوات</h1>

      {/* Voice Picker (dropdown style — like the design from test.md) */}
      <div className="bg-card border border-border rounded-xl p-6 space-y-4">
        <label className="text-sm text-muted-foreground">اختيار الصوت</label>
        <VoicePicker
          voices={voices}
          value={selected}
          onValueChange={(v) => { setSelected(v); setPickerOpen(true) }}
          open={pickerOpen}
          onOpenChange={setPickerOpen}
          placeholder="اختر صوتاً..."
        />
      </div>

      {/* Voice list grid (alternative selection view) */}
      <div className="bg-card border border-border rounded-xl p-6 space-y-3">
        <input
          type="text" value={search} onChange={(e) => setSearch(e.target.value)}
          placeholder="🔍 بحث..."
          className="w-full px-3 py-2 rounded-md border border-border bg-secondary text-foreground"
        />
        <div className="space-y-2">
          {filtered.length === 0 && <div className="text-center text-muted-foreground py-8">لا توجد أصوات</div>}
          {filtered.map(v => {
            const lang = v.labels?.language || ""
            const langLabel = { ar: "العربية", en: "English", multi: "متعدد" }[lang] || lang
            return (
              <div
                key={v.voiceId}
                onClick={() => setSelected(v.voiceId)}
                className={cn(
                  "flex items-center gap-3 p-3 rounded-lg border cursor-pointer transition-colors",
                  v.voiceId === selected ? "border-primary bg-primary/10" : "border-border bg-secondary hover:bg-accent"
                )}
              >
                <div
                  className="size-10 shrink-0 rounded-full overflow-visible relative cursor-pointer"
                  onClick={(e) => { e.stopPropagation(); previewVoice(v) }}
                >
                  <Orb className="absolute inset-0" />
                </div>
                <div className="flex-1">
                  <div className="font-medium">{v.name}</div>
                  <div className="text-xs text-muted-foreground">
                    <span className="inline-block px-2 py-0.5 rounded bg-green-900/40 text-green-400 mr-2">{langLabel}</span>
                  </div>
                </div>
                {v.voiceId !== "default" && (
                  <Button
                    variant="ghost" size="icon"
                    onClick={(e) => { e.stopPropagation(); deleteVoice(v) }}
                  >
                    <Trash2 className="size-4" />
                  </Button>
                )}
              </div>
            )
          })}
        </div>
      </div>

      {/* Player */}
      {selectedVoice && (
        <div className="bg-card border border-border rounded-xl p-6 space-y-3 sticky bottom-4">
          <label className="text-sm text-muted-foreground">
            اكتب نصاً ليُقرأ بصوت <span className="text-foreground font-semibold">{selectedVoice.name}</span>
          </label>
          <textarea
            value={text} onChange={(e) => setText(e.target.value)}
            className="w-full min-h-[90px] px-3 py-2 rounded-md border border-border bg-secondary text-foreground resize-y"
            dir="rtl"
          />
          <div className="flex items-center gap-3">
            <Button onClick={play} disabled={isPlaying}>
              <Volume2 className="mr-2 size-4" />تشغيل
            </Button>
            <Button onClick={stop} disabled={!isPlaying} variant="outline">
              <VolumeX className="mr-2 size-4" />إيقاف
            </Button>
            <span className={cn("text-sm font-mono ml-auto", ttfa && ttfa < 500 && "text-green-400 font-semibold")}>
              {ttfa !== null ? `TTFA: ${ttfa}ms` : status}
            </span>
          </div>
        </div>
      )}
    </div>
  )
}
