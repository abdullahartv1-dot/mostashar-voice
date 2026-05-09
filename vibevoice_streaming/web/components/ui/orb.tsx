"use client"

import * as React from "react"
import { cn } from "@/lib/utils"

interface OrbProps {
  agentState?: "thinking" | "talking" | undefined
  className?: string
}

export function Orb({ agentState, className }: OrbProps) {
  return (
    <div className={cn("relative w-full h-full rounded-full overflow-hidden", className)}>
      <div
        className={cn(
          "absolute inset-0 rounded-full",
          agentState === "thinking" && "orb-rotate",
          agentState === "talking" && "orb-pulse"
        )}
        style={{
          background:
            "conic-gradient(from 0deg, #4f8cff, #b14fff, #ff4f9c, #ffb84f, #4fffc8, #4f8cff)",
        }}
      />
      <div
        className="absolute inset-[8%] rounded-full"
        style={{
          background:
            "radial-gradient(circle at 30% 30%, rgba(255,255,255,0.35), transparent 60%), radial-gradient(circle at 70% 70%, rgba(0,0,0,0.4), transparent 60%)",
        }}
      />
    </div>
  )
}
