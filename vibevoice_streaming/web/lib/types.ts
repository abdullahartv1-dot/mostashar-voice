// Shape compatible with the ElevenLabs.Voice type used by the original VoicePicker.
// Maps from our backend's GET /voices response.
export interface Voice {
  voiceId: string
  name: string
  category?: string
  description?: string
  previewUrl?: string
  labels?: {
    accent?: string
    description?: string
    descriptive?: string
    age?: string
    gender?: string
    language?: string
    use_case?: string
  }
}

export interface BackendVoice {
  voice_id: string
  name: string
  language: string
  dur_s: number
  preview_url: string
  created_at: number | null
}

export function backendToVoice(v: BackendVoice): Voice {
  const langLabel = { ar: "العربية", en: "English", multi: "Multilingual" }[v.language] || v.language
  return {
    voiceId: v.voice_id,
    name: v.name,
    category: "user",
    previewUrl: v.preview_url,
    labels: {
      language: v.language,
      accent: langLabel,
    },
  }
}
