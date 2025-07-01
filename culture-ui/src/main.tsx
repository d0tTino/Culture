import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import './index.css'
import App from './App.tsx'
import { widgetRegistry } from './lib/widgetRegistry'
import { TimelineWidget, BreakpointList, EventConsole } from './widgets'

widgetRegistry.register('Timeline', TimelineWidget)
widgetRegistry.register('Breakpoints', BreakpointList)
widgetRegistry.register('Events', EventConsole)

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </StrictMode>,
)
