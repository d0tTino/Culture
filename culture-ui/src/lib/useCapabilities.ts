import { useEffect, useState } from 'react'
import { fetchCapabilities, type CapabilityState } from './api'

export function useCapabilities() {
  const [capabilities, setCapabilities] = useState<Record<string, CapabilityState>>({})

  useEffect(() => {
    let cancelled = false
    async function load() {
      try {
        const envelope = await fetchCapabilities()
        if (!cancelled) {
          setCapabilities(envelope.data.capabilities || {})
        }
      } catch {
        if (!cancelled) {
          setCapabilities({})
        }
      }
    }
    void load()
    return () => {
      cancelled = true
    }
  }, [])

  return capabilities
}
