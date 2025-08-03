import { useEffect, useState } from 'react'

interface AgentInfo {
  mood?: number
  summary?: string
}

interface MessagePayload {
  agent_id?: string
  content?: string
}

export default function MemoryExplorerPage() {
  const [agents, setAgents] = useState<string[]>([])
  const [agentId, setAgentId] = useState('')
  const [semantic, setSemantic] = useState<string[]>([])
  const [episodic, setEpisodic] = useState<string[]>([])
  const [messages, setMessages] = useState<string[]>([])

  useEffect(() => {
    let cancelled = false
    async function load() {
      try {
        const res = await fetch('/api/map')
        const json = (await res.json()) as {
          agents?: Record<string, AgentInfo>
        }
        if (cancelled) return
        const ids = Object.keys(json.agents || {})
        setAgents(ids)
        if (ids.length) setAgentId(ids[0])
      } catch {
        /* ignore */
      }
    }
    void load()
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    if (!agentId) return
    let cancelled = false
    async function load() {
      try {
        const res = await fetch(`/api/memory/${agentId}`)
        const json = (await res.json()) as {
          semantic?: string[]
          episodic?: Array<{ content?: string } | string>
        }
        if (cancelled) return
        setSemantic(json.semantic || [])
        const eps = (json.episodic || []).map((m) =>
          typeof m === 'string' ? m : m.content || String(m),
        )
        setEpisodic(eps)
      } catch {
        /* ignore */
      }
    }
    void load()
    return () => {
      cancelled = true
    }
  }, [agentId])

  useEffect(() => {
    if (!agentId) return
    const es = new EventSource('/stream/messages')
    es.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data) as MessagePayload
        if (!msg.agent_id || msg.agent_id === agentId) {
          setMessages((cur) => [...cur, msg.content || String(ev.data)])
        }
      } catch {
        setMessages((cur) => [...cur, String(ev.data)])
      }
    }
    return () => es.close()
  }, [agentId])

  return (
    <div className="p-4 space-y-4">
      <h1 className="text-xl font-bold">Memory Explorer</h1>
      <label className="block">
        Agent:
        <select
          aria-label="agent-select"
          value={agentId}
          onChange={(e) => setAgentId(e.target.value)}
          className="border p-1 ml-2"
        >
          {agents.map((id) => (
            <option key={id} value={id}>
              {id}
            </option>
          ))}
        </select>
      </label>
      <div data-testid="summaries">
        {semantic.map((s, i) => (
          <div key={i}>{s}</div>
        ))}
      </div>
      <div data-testid="memories">
        {episodic.map((m, i) => (
          <div key={i}>{m}</div>
        ))}
      </div>
      <div data-testid="messages">
        {messages.map((m, i) => (
          <div key={i}>{m}</div>
        ))}
      </div>
    </div>
  )
}

