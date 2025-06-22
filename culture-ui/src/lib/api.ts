export interface Mission {
  id: number
  name: string
  status: string
  progress: number
}

export async function fetchMissions(): Promise<Mission[]> {
  const response = await fetch('/api/missions')
  if (!response.ok) {
    throw new Error('Failed to fetch missions')
  }
  return response.json()
}
