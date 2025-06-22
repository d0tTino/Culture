import { render, screen } from '@testing-library/react'
import WorldMap from './widgets/WorldMap'

describe('WorldMap', () => {
  it('renders canvas placeholder', () => {
    render(<WorldMap />)
    expect(screen.getByTestId('world-map-canvas')).toBeInTheDocument()
  })
})
