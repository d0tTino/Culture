import { useState, FormEvent } from 'react'
import { registerWidget } from '../lib/widgetRegistry'
import { proposeLaw } from '../lib/api'

export default function LawProposal() {
  const [text, setText] = useState('')
  const [proposer, setProposer] = useState('agent_1')
  const [result, setResult] = useState<string | null>(null)

  async function submit(e: FormEvent) {
    e.preventDefault()
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
      <form onSubmit={submit} className="space-y-2">
        <input
          placeholder="Proposal text"
          value={text}
          onChange={(e) => setText(e.target.value)}
          className="border p-1 w-full"
        />
        <input
          placeholder="Proposer ID"
          value={proposer}
          onChange={(e) => setProposer(e.target.value)}
          className="border p-1 w-full"
        />
        <button type="submit" className="px-2 py-1 border rounded">
          Submit
        </button>
      </form>
      {result && <div data-testid="result">Result: {result}</div>}
    </div>
  )
}

registerWidget('LawProposal', LawProposal)
