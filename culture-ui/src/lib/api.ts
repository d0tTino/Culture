export interface Mission {
  id: number
  name: string
  status: string
  progress: number
}

export async function fetchMissions(): Promise<Mission[]> {
  const res = await fetch('/api/missions')
  return (await res.json()) as Mission[]
}

