import { useMemo, useEffect, useRef } from 'react'
import ForceGraph2D from 'react-force-graph-2d'

export default function NetworkWeb() {
  const data = useMemo(
    () => ({
      nodes: [
        { id: 'Agent1' },
        { id: 'Agent2' },
        { id: 'Agent3' },
      ],
      links: [
        { source: 'Agent1', target: 'Agent2' },
        { source: 'Agent2', target: 'Agent3' },
      ],
    }),
    [],
  )

  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const el = containerRef.current
    if (el) el.style.pointerEvents = 'none'
    const canvas = el?.querySelector('canvas')
    if (canvas) canvas.style.pointerEvents = 'none'
  }, [])

  return (
    <div ref={containerRef} style={{ width: '100%', height: 400 }}>
      <ForceGraph2D graphData={data} enablePointerInteraction={false} />
    </div>
  )
}
