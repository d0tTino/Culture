import { render, screen } from '@testing-library/react'
import LawProposal from './pages/LawProposal'
import userEvent from '@testing-library/user-event'
import { vi } from 'vitest'

afterEach(() => {
  vi.unstubAllGlobals()
})

test('submits proposal and shows result', async () => {
  const user = userEvent.setup()
  vi.stubGlobal('fetch', vi.fn(() =>
    Promise.resolve({
      json: () => Promise.resolve({ approved: true })
    }) as unknown as Response
  ))

  render(<LawProposal />)
  await user.type(screen.getByPlaceholderText('Proposal text'), 'L')
  await user.type(screen.getByPlaceholderText('Proposer ID'), 'A')
  await user.click(screen.getByRole('button', { name: /submit/i }))
  expect(await screen.findByTestId('result')).toHaveTextContent('Approved')
})
