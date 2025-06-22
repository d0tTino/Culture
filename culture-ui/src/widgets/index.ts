export { registerWidget, getWidget } from './WidgetRegistry'

import NetworkWeb from './NetworkWeb'
import WorldMap from './WorldMap'
import { registerWidget } from './WidgetRegistry'

export function registerBuiltins() {
  registerWidget('NetworkWeb', NetworkWeb)
  registerWidget('WorldMap', WorldMap)
}

export { NetworkWeb, WorldMap }
