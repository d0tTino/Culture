import type { IJsonModel } from 'flexlayout-react'
import { listWidgets } from './widgetRegistry'

export function createDefaultLayout(): IJsonModel {
  const widgets = listWidgets()
  const order = ['NetworkWeb', 'WorldMap', 'TimelineWidget', 'KpiCard', 'MemoryExplorer']
  const sorted = [
    ...order.filter((n) => widgets.includes(n)),
    ...widgets.filter((n) => !order.includes(n)),
  ]

  return {
    global: {},
    layout: {
      type: 'row',
      children: [
        {
          type: 'tabset',
          children: sorted.map((name) => ({
            type: 'tab',
            name,
            component: name,
          })),
        },
      ],
    },
  }
}
