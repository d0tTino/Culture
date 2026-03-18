import KpiCard from '../widgets/KpiCard'
import { registerWidget } from '../lib/widgetRegistry'

const KPI_DOCS = [
  {
    name: 'payload_version',
    meaning: 'Schema version for /api/user_value_metrics. Increment when KPI semantics or field contracts change.',
  },
  {
    name: 'narrative_continuity_score',
    meaning: 'Average of knowledge-board continuation links and contiguous event steps; higher means the story progresses coherently across turns.',
  },
  {
    name: 'unresolved_conflict_count',
    meaning: 'Conflicts detected on the knowledge board that do not yet have a linked resolution entry.',
  },
  {
    name: 'cross_agent_interaction_diversity',
    meaning: 'Observed directed agent-to-agent interaction pairs divided by the total possible directed pairs.',
  },
  {
    name: 'user_intervention_rate',
    meaning: 'Share of logged simulation events that were initiated by human commands.',
  },
  {
    name: 'return_session_continuity',
    meaning: 'Continuity of snapshot/resume progression across recorded snapshot steps.',
  },
  {
    name: 'novelty_score',
    meaning: 'Unique action intents divided by total action intents for the sampled window.',
  },
  {
    name: 'repetitive_intents_ratio',
    meaning: 'Frequency of the most common action intent divided by total action intents; higher means more repetition.',
  },
  {
    name: 'social_graph_change_count',
    meaning: 'Count of events whose payload indicates relationship, coalition, ally, or rival changes.',
  },
  {
    name: 'stagnation_alerts',
    meaning: 'Independent alerts emitted when novelty, interaction diversity, repetitive intent, or social-graph metrics cross their dedicated thresholds.',
  },
]

export default function KpiCardPage() {
  return (
    <div className="p-4 space-y-4">
      <h1 className="text-xl font-bold">KPI Card</h1>
      <p className="max-w-3xl text-sm text-gray-700">
        Canonical KPI definitions live here and in the repository README so product and
        engineering interpret the dashboard consistently.
      </p>
      <KpiCard />
      <section aria-labelledby="kpi-definitions" className="max-w-4xl space-y-3">
        <h2 id="kpi-definitions" className="text-lg font-semibold">
          User value KPI definitions
        </h2>
        <div className="overflow-x-auto rounded border">
          <table className="min-w-full border-collapse text-left text-sm">
            <thead className="bg-gray-50">
              <tr>
                <th className="border-b px-3 py-2 font-semibold">Field</th>
                <th className="border-b px-3 py-2 font-semibold">Definition</th>
              </tr>
            </thead>
            <tbody>
              {KPI_DOCS.map((doc) => (
                <tr key={doc.name}>
                  <td className="border-b px-3 py-2 font-mono">{doc.name}</td>
                  <td className="border-b px-3 py-2">{doc.meaning}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  )
}

registerWidget('KpiCard', KpiCardPage)
