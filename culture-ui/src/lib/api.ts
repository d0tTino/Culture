import type { WidgetInfo } from './widgetRegistry'

export interface ApiEnvelope<T> {
  schema: string
  version: string
  enabled: boolean
  data: T
  fallback?: {
    reason?: string
    action?: string
  }
}

export interface CapabilityState {
  enabled: boolean
  reason: string
}

export interface CapabilitiesResponse {
  capabilities: Record<string, CapabilityState>
}

export interface Mission {
  id: number
  name: string
  status: string
  progress: number
}

export interface Quest {
  id: number
  title: string
  description: string
  progress: number
  status: string
}

export interface ProposalOutcome {
  approved: boolean
  yes_weight: number
  no_weight: number
  ip_spent: number
}

export interface ObservabilityMetrics {
  du_per_1k_tokens: number
  llm_latency_p95_ms: number
  coalition_count: number
  average_sentiment: number
  rag_hit_rate: number
  memory_retrievals_total: number
  memory_retrieval_errors_total: number
  memory_retrieval_success_rate: number
  memory_retrieval_error_rate: number
  llm_errors_total: number
  llm_error_rate: number
  agent_du_per_1k_tokens?: Record<string, number>
  agent_llm_latency_p95_ms?: Record<string, number>
}

export type WidgetRegistration = WidgetInfo & Record<string, unknown>

function isEnvelope<T>(value: unknown): value is ApiEnvelope<T> {
  return typeof value === 'object' && value !== null && 'schema' in value && 'data' in value
}

export async function fetchApiEnvelope<T>(url: string): Promise<ApiEnvelope<T>> {
  const res = await fetch(url)
  const json = (await res.json()) as ApiEnvelope<T> | T
  if (isEnvelope<T>(json)) {
    return json
  }
  return {
    schema: url,
    version: 'legacy',
    enabled: true,
    data: json as T,
  }
}

export async function fetchCapabilities(): Promise<ApiEnvelope<CapabilitiesResponse>> {
  return fetchApiEnvelope<CapabilitiesResponse>('/api/capabilities')
}

export async function fetchMissions(): Promise<ApiEnvelope<{ missions: Mission[] }>> {
  return fetchApiEnvelope<{ missions: Mission[] }>('/api/missions')
}

export async function fetchQuests(): Promise<ApiEnvelope<{ quests: Quest[] }>> {
  return fetchApiEnvelope<{ quests: Quest[] }>('/api/quests')
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

export async function submitProposal(
  proposerId: string,
  text: string,
): Promise<ProposalOutcome> {
  const res = await fetch('/api/governance/propose', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ proposer_id: proposerId, text }),
  })
  return (await res.json()) as ProposalOutcome
}

export async function registerWidgetBackend(widget: WidgetRegistration): Promise<string[]> {
  const res = await fetch('/api/register_widget', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      name: widget.name,
      ...(widget.scriptUrl ? { script_url: widget.scriptUrl } : {}),
      ...Object.fromEntries(
        Object.entries(widget).filter(([k]) => k !== 'name' && k !== 'scriptUrl'),
      ),
    }),
  })
  const data = (await res.json()) as { widgets: string[] }
  return data.widgets
}

export async function fetchObservabilityMetrics(): Promise<ApiEnvelope<ObservabilityMetrics>> {
  return fetchApiEnvelope<ObservabilityMetrics>('/api/observability_metrics')
}

export async function displayObservabilityMetrics(): Promise<void> {
  const envelope = await fetchObservabilityMetrics()
  const metrics = envelope.data
  console.log('DU/1k tokens:', metrics.du_per_1k_tokens)
  console.log('p95 latency (ms):', metrics.llm_latency_p95_ms)
  console.log('Coalitions:', metrics.coalition_count)
  console.log('Average sentiment:', metrics.average_sentiment)
  console.log('RAG hit rate:', metrics.rag_hit_rate)
  console.log('Memory retrievals (total):', metrics.memory_retrievals_total)
  console.log('Memory retrieval errors (total):', metrics.memory_retrieval_errors_total)
  console.log('Memory retrieval success rate:', metrics.memory_retrieval_success_rate)
  console.log('Memory retrieval error rate:', metrics.memory_retrieval_error_rate)
  console.log('LLM errors (total):', metrics.llm_errors_total)
  console.log('LLM error rate:', metrics.llm_error_rate)
  console.log('Agent DU/1k tokens:', metrics.agent_du_per_1k_tokens)
  console.log('Agent p95 latency (ms):', metrics.agent_llm_latency_p95_ms)
}
