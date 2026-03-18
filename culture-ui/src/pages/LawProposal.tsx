import { useState } from 'react'
import type { FormEvent } from 'react'
import { registerWidget } from '../lib/widgetRegistry'
import { proposeLaw } from '../lib/api'
import { useCapabilities } from '../lib/useCapabilities'

export default function LawProposal() {
  const capabilities = useCapabilities()
  const governance = capabilities.governance
  const [text, setText] = useState('')
  const [proposer, setProposer] = useState('agent_1')
  const [result, setResult] = useState<string | null>(null)

  async function submit(e: FormEvent) {
    e.preventDefault()
    if (governance && !governance.enabled) {
      setResult(governance.reason)
      return
    }
    setResult(null)
    try {
      const approved = await proposeLaw(proposer, text)
      setResult(approved ? 'Approved' : 'Rejected')
    } catch {
      setResult('Error')
    }
  }

  return (
    <div className="p-4 space-y-4">
      <h1 className="text-xl font-bold">Propose Law</h1>
      {governance && !governance.enabled && <p data-testid="governance-fallback">{governance.reason}</p>}
      <form onSubmit={submit} className="space-y-2">
        <input
          placeholder="Proposal text"
          value={text}
          onChange={(e) => setText(e.target.value)}
          className="border p-1 w-full"
          disabled={Boolean(governance && !governance.enabled)}
        />
        <input
          placeholder="Proposer ID"
          value={proposer}
          onChange={(e) => setProposer(e.target.value)}
          className="border p-1 w-full"
          disabled={Boolean(governance && !governance.enabled)}
        />
        <button type="submit" className="px-2 py-1 border rounded" disabled={Boolean(governance && !governance.enabled)}>
          Submit
        </button>
      </form>
      {result && <div data-testid="result">Result: {result}</div>}
    </div>
  )
}

registerWidget('LawProposal', LawProposal)
