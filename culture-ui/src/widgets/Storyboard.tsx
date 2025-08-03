import { useEffect, useState } from 'react'
import { useEventSource } from '../lib/useEventSource'
import Heatmap from '../components/Heatmap'
import MemoryTimeline from '../components/MemoryTimeline'

interface SnapshotEvent {
  type?: string
  data?: {
    world_map?: { agents?: Record<string, [number, number]> }
    agents?: Array<{ agent_id: string; mood?: number }>
    step?: number
  }
}

interface AgentState {
  state?: {
    ip?: string
  }
}

export default function Storyboard() {
  const event = useEventSource<SnapshotEvent>()
  const [positions, setPositions] = useState<Record<string, [number, number]>>({})
  const [moods, setMoods] = useState<Record<string, number>>({})
  const [retrievals, setRetrievals] = useState<Record<string, number>>({})
  const [agentStates, setAgentStates] = useState<Record<string, AgentState>>({})
  const [heatmap, setHeatmap] = useState<Record<string, number>>({})
  const [memoryEvents, setMemoryEvents] = useState<
    Array<{ type: string; step?: number }>
  >([])
  const [tab, setTab] = useState<'map' | 'summaries'>('map')
  const [summaries, setSummaries] = useState<string[]>([])

  useEffect(() => {
    let cancelled = false
    async function load() {
      try {
        const res = await fetch('/api/agent_stats')
        const json = (await res.json()) as {
          agents?: Record<string, { mood?: number; retrieval_count?: number }>
        }
        if (json.agents && !cancelled) {
          const counts: Record<string, number> = {}
          const states: Record<string, AgentState> = {}
          await Promise.all(
            Object.entries(json.agents).map(async ([id, info]) => {
              if (typeof info.retrieval_count === 'number') {
                counts[id] = info.retrieval_count
              }
              if (typeof info.mood === 'number') {
                setMoods((cur) => ({ ...cur, [id]: info.mood! }))
              }
              try {
                const sres = await fetch(`/api/agents/${id}/state`)
                states[id] = (await sres.json()) as AgentState
              } catch {
                /* ignore */
              }
            }),
          )
          setRetrievals(counts)
          setAgentStates((cur) => ({ ...cur, ...states }))
        }
      } catch {
        /* ignore */
      }
    }
    void load()
    return () => {
      cancelled = true
    }
  }, [event])

  const agentId = Object.keys(positions)[0] || Object.keys(moods)[0] || 'agent-1'

  useEffect(() => {
    if (event?.data?.world_map?.agents) {
      setPositions(event.data.world_map.agents)
      setHeatmap((cur) => {
        const copy = { ...cur }
        for (const [, pos] of Object.entries(event.data!.world_map!.agents!)) {
          const key = `${Math.round(pos[0])},${Math.round(pos[1])}`
          copy[key] = (copy[key] || 0) + 1
        }
        return copy
      })
    }
    if (event?.data?.agents) {
      const m: Record<string, number> = {}
      for (const a of event.data.agents) {
        if (typeof a.mood === 'number') m[a.agent_id] = a.mood
      }
      setMoods((cur) => ({ ...cur, ...m }))
    }
    if (event?.type?.startsWith('memory')) {
      setMemoryEvents((cur) =>
        [{ type: event.type!, step: event.data?.step }, ...cur].slice(0, 20),
      )
    }
  }, [event])

  useEffect(() => {
    if (tab !== 'summaries') return
    let cancelled = false
    async function load() {
      try {
        const res = await fetch(`/api/agents/${agentId}/semantic_summaries`)
        const json = await res.json()
        if (!cancelled) setSummaries(json.summaries || [])
      } catch {
        /* ignore */
      }
    }
    void load()
    return () => {
      cancelled = true
    }
  }, [tab, agentId])

  return (
    <div className="p-2" data-testid="storyboard">
      <div className="space-x-2 mb-2">
        <button onClick={() => setTab('map')}>Map</button>
        <button onClick={() => setTab('summaries')}>Summaries</button>
      </div>
      {tab === 'map' ? (
        <div>
          <ul data-testid="map-info" className="mb-2">
            {Object.entries(positions).map(([id, pos]) => (
              <li key={id}>
                {id}: {pos[0]}, {pos[1]} (mood {moods[id] ?? 'n/a'}, ip{' '}
                {agentStates[id]?.state?.ip ?? 'n/a'}, retrievals {retrievals[id] ?? 0})
              </li>
            ))}
          </ul>
          <Heatmap data={heatmap} />
        </div>
      ) : (
        <div data-testid="summaries">
          {summaries.map((s, i) => (
            <div key={i}>{s}</div>
          ))}
        </div>
      )}
      <MemoryTimeline events={memoryEvents} />
    </div>
  )
}
