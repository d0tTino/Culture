import type { IJsonModel } from 'flexlayout-react'
import { listWidgets } from './widgetRegistry'

export function createDefaultLayout(): IJsonModel {
  return {
    global: {},
    layout: {
      type: 'row',
      children: [
        {
          type: 'tabset',
          children: listWidgets().map((name) => ({
            type: 'tab',
            name,
            component: name,
          })),
        },
      ],
    },
  }
}
