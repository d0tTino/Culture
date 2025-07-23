
export interface HeatmapProps {
  data: Record<string, number>
  size?: number
  gridSize?: number
}

export default function Heatmap({ data, size = 10, gridSize = 10 }: HeatmapProps) {
  const cells = Array.from({ length: gridSize * gridSize })
  const max = Math.max(1, ...Object.values(data))
  return (
    <div
      data-testid="heatmap"
      className="grid border"
      style={{ gridTemplateColumns: `repeat(${gridSize}, 1fr)`, width: size * 4, height: size * 4 }}
    >
      {cells.map((_, i) => {
        const x = i % gridSize
        const y = Math.floor(i / gridSize)
        const count = data[`${x},${y}`] || 0
        const alpha = count / max
        return (
          <div key={`${x}-${y}`} style={{ backgroundColor: `rgba(255, 0, 0, ${alpha})` }} />
        )
      })}
    </div>
  )
}
