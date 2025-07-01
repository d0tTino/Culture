import { useEffect, useState } from 'react'

export default function HeaderBar() {
  const [time, setTime] = useState(() => new Date())
  const [paused, setPaused] = useState(false)
  const [speed, setSpeed] = useState(1)
  const scenarios = ['Demo', 'Test']

  useEffect(() => {
    const id = setInterval(() => setTime(new Date()), 1000)
    return () => clearInterval(id)
  }, [])

  const sendCommand = async (cmd: Record<string, unknown>) => {
    await fetch('/control', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(cmd),
    })
  }

  const togglePause = async () => {
    const command = paused ? 'resume' : 'pause'
    await sendCommand({ command })
    setPaused(!paused)
  }

  const handleSpeedChange = async (
    e: React.ChangeEvent<HTMLInputElement>,
  ) => {
    const value = Number(e.target.value)
    setSpeed(value)
    await sendCommand({ command: 'set_speed', speed: value })
  }

  return (
    <header className="flex items-center justify-between p-2 bg-gray-200">
      <div>{time.toLocaleTimeString()}</div>
      <div className="flex items-center space-x-2">
        <label className="flex items-center space-x-1">
          <span>Speed</span>
          <input
            aria-label="speed"
            type="range"
            min="0.1"
            max="5"
            step="0.1"
            value={speed}
            onChange={handleSpeedChange}
          />
        </label>
        <button onClick={togglePause}>{paused ? 'Resume' : 'Pause'}</button>
        <select aria-label="scenario">
          {scenarios.map((s) => (
            <option key={s}>{s}</option>
          ))}
        </select>
      </div>
    </header>
  )
}
