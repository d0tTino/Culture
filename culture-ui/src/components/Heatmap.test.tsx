import { render, screen } from '@testing-library/react'
import Heatmap from './Heatmap'

describe('Heatmap', () => {
  it('renders cells with alpha based on count', () => {
    render(<Heatmap data={{ '0,0': 5 }} />)
    const firstCell = screen.getByTestId('heatmap').firstChild as HTMLElement
    expect(firstCell.style.backgroundColor).toBe('rgb(255, 0, 0)')
  })
})
