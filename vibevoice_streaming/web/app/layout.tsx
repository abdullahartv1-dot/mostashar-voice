import type { Metadata } from "next"
import "./globals.css"
import { NavBar } from "@/components/nav-bar"

export const metadata: Metadata = {
  title: "مُسْتَشَار — استنساخ الصوت",
  description: "Voice cloning + realtime TTS",
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ar" dir="rtl">
      <body>
        <NavBar />
        <main className="max-w-4xl mx-auto px-6 py-8">{children}</main>
      </body>
    </html>
  )
}
