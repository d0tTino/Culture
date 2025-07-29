import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import './index.css'
import App from './App.tsx'
import { widgetRegistry, loadRemoteWidgets } from './lib/widgetRegistry'
import {
  TimelineWidget,
  BreakpointList,
  NetworkWeb,
  WorldMap,
  KpiCard,
  EventConsole,
  Storyboard,
  LiveMap,
  ResourceHistory,
} from './widgets'

widgetRegistry.register('TimelineWidget', TimelineWidget)
widgetRegistry.register('NetworkWeb', NetworkWeb)
widgetRegistry.register('WorldMap', WorldMap)
widgetRegistry.register('KpiCard', KpiCard)
widgetRegistry.register('LiveMap', LiveMap)
widgetRegistry.register('ResourceHistory', ResourceHistory)
widgetRegistry.register('Breakpoints', BreakpointList)
widgetRegistry.register('Events', EventConsole)
widgetRegistry.register('Storyboard', Storyboard)

void loadRemoteWidgets();

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </StrictMode>,
)
