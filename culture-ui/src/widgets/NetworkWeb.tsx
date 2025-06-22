import ForceGraph2D from 'react-force-graph-2d'
import { useMemo } from 'react'

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

  return (
    <div style={{ width: '100%', height: 400 }}>
      <ForceGraph2D graphData={data} />
    </div>
  )
}
