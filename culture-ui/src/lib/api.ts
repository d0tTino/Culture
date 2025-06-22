export interface Mission {
  id: number
  name: string
  status: string
  progress: number
}

const API_BASE = ''

export async function fetchMissions(): Promise<Mission[]> {
  const response = await fetch(`${API_BASE}/api/missions`)
  if (!response.ok) {
    throw new Error('Failed to fetch missions')
  }
  return (await response.json()) as Mission[]
}
