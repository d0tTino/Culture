import DockManager from './components/DockManager'
import Home from './pages/Home'
import MissionOverview from './pages/MissionOverview'
import AgentDataOverview from './pages/AgentDataOverview'
import { registerWidget } from './lib/widgetRegistry'

const defaultLayout = {
  global: { tabEnableClose: false },
  layout: {
    type: 'row',
    weight: 100,
    children: [
      {
        type: 'tabset',
        weight: 30,
        children: [{ type: 'tab', name: 'Home', component: 'home' }],
      },
      {
        type: 'tabset',
        weight: 70,
        selected: 0,
        children: [
          { type: 'tab', name: 'Mission Overview', component: 'missions' },
          { type: 'tab', name: 'Agent Data', component: 'agentData' },
        ],
      },
    ],
  },
}

registerWidget('home', Home)
registerWidget('missions', MissionOverview)
registerWidget('agentData', AgentDataOverview)

export default function App() {
  return <DockManager defaultLayout={defaultLayout} />
}
