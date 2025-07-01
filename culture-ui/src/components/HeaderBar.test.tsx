import { render, screen, fireEvent } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, vi, afterEach } from 'vitest'
import HeaderBar from './HeaderBar'

afterEach(() => {
  vi.restoreAllMocks()
})

describe('HeaderBar', () => {
  it('toggles pause and resume', async () => {
    const fetchMock = vi.fn(() => Promise.resolve({ ok: true } as Response))
    global.fetch = fetchMock as unknown as typeof fetch
    render(<HeaderBar />)
    const button = screen.getByRole('button', { name: /pause/i })
    await userEvent.click(button)
    expect(fetchMock).toHaveBeenCalledWith(
      '/control',
      expect.objectContaining({
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ command: 'pause' }),
      }),
    )
    await userEvent.click(button)
    expect(fetchMock).toHaveBeenCalledWith(
      '/control',
      expect.objectContaining({
        body: JSON.stringify({ command: 'resume' }),
      }),
    )
  })

  it('sends speed updates', () => {
    const fetchMock = vi.fn(() => Promise.resolve({ ok: true } as Response))
    global.fetch = fetchMock as unknown as typeof fetch
    render(<HeaderBar />)
    const slider = screen.getByRole('slider') as HTMLInputElement
    fireEvent.change(slider, { target: { value: '2' } })
    expect(fetchMock).toHaveBeenCalledWith(
      '/control',
      expect.objectContaining({
        body: JSON.stringify({ command: 'set_speed', speed: 2 }),
      }),
    )
  })
})
