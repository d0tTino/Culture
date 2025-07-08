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

export async function proposeLaw(
  proposerId: string,
  text: string,
): Promise<boolean> {
  const res = await fetch('/api/propose_law', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ proposer_id: proposerId, text }),
  })
  const data = (await res.json()) as { approved: boolean }
  return data.approved
}

