import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import Header from './Header'

describe('Header theme toggle', () => {
  beforeEach(() => {
    document.body.className = ''
    localStorage.clear()
  })

  it('toggles dark class on body', async () => {
    const user = userEvent.setup()
    render(<Header />)
    const button = screen.getByRole('button', { name: /toggle theme/i })

    expect(document.body.classList.contains('dark')).toBe(false)

    await user.click(button)
    expect(document.body.classList.contains('dark')).toBe(true)
    expect(localStorage.getItem('theme')).toBe('dark')

    await user.click(button)
    expect(document.body.classList.contains('dark')).toBe(false)
    expect(localStorage.getItem('theme')).toBe('light')
  })
})
