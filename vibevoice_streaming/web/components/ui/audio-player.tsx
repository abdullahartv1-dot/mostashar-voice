"use client"

import * as React from "react"

interface AudioItem {
  id: string
  src: string
  data?: any
}

interface AudioPlayerCtx {
  current: AudioItem | null
  isPlaying: boolean
  isItemActive: (id: string) => boolean
  play: (item: AudioItem) => void
  pause: () => void
}

const Ctx = React.createContext<AudioPlayerCtx | null>(null)

export function AudioPlayerProvider({ children }: { children: React.ReactNode }) {
  const audioRef = React.useRef<HTMLAudioElement | null>(null)
  const [current, setCurrent] = React.useState<AudioItem | null>(null)
  const [isPlaying, setIsPlaying] = React.useState(false)

  React.useEffect(() => {
    audioRef.current = new Audio()
    const a = audioRef.current
    const onPlay = () => setIsPlaying(true)
    const onPause = () => setIsPlaying(false)
    const onEnded = () => { setIsPlaying(false); setCurrent(null) }
    a.addEventListener("play", onPlay)
    a.addEventListener("pause", onPause)
    a.addEventListener("ended", onEnded)
    return () => {
      a.pause()
      a.removeEventListener("play", onPlay)
      a.removeEventListener("pause", onPause)
      a.removeEventListener("ended", onEnded)
    }
  }, [])

  const play = React.useCallback((item: AudioItem) => {
    const a = audioRef.current
    if (!a) return
    if (current?.id !== item.id) {
      a.src = item.src
      setCurrent(item)
    }
    a.play().catch(() => {})
  }, [current])

  const pause = React.useCallback(() => audioRef.current?.pause(), [])

  const isItemActive = React.useCallback((id: string) => current?.id === id, [current])

  return (
    <Ctx.Provider value={{ current, isPlaying, isItemActive, play, pause }}>
      {children}
    </Ctx.Provider>
  )
}

export function useAudioPlayer() {
  const v = React.useContext(Ctx)
  if (!v) throw new Error("useAudioPlayer must be inside AudioPlayerProvider")
  return v
}
