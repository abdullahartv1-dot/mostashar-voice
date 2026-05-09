"use client"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { cn } from "@/lib/utils"

export function NavBar() {
  const path = usePathname()
  return (
    <nav className="border-b border-border bg-card">
      <div className="max-w-4xl mx-auto px-6 flex items-center">
        <Link href="/clone" className={cn("py-4 px-5 font-medium border-b-2 transition-colors",
          path === "/clone" || path === "/" ? "border-primary text-foreground" : "border-transparent text-muted-foreground hover:text-foreground"
        )}>
          📤 استنساخ صوت
        </Link>
        <Link href="/library" className={cn("py-4 px-5 font-medium border-b-2 transition-colors",
          path === "/library" ? "border-primary text-foreground" : "border-transparent text-muted-foreground hover:text-foreground"
        )}>
          🎤 مكتبة الأصوات
        </Link>
        <div className="flex-1" />
        <span className="text-xs text-muted-foreground py-4">VibeVoice-Large + 7B</span>
      </div>
    </nav>
  )
}
