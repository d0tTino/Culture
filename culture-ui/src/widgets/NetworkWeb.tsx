import { useMemo, type FC } from 'react'

let ForceGraph2D: FC<Record<string, unknown>> = () => <canvas />
if (process.env.NODE_ENV !== 'test') {
  // dynamic import avoids bundling in tests
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  ForceGraph2D = require('react-force-graph-2d').default
}

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
