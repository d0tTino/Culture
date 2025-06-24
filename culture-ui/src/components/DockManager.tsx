import { Layout, Model, TabNode, type IJsonModel } from 'flexlayout-react'
import { useState, useCallback } from 'react'
import { getWidget, listWidgets } from '../lib/widgetRegistry'
import 'flexlayout-react/style/light.css'

export interface DockManagerProps {
  defaultLayout: IJsonModel
}

const STORAGE_KEY = 'dockLayout'

export default function DockManager({ defaultLayout }: DockManagerProps) {
  const [model] = useState(() => {
    const saved = localStorage.getItem(STORAGE_KEY)
    if (saved) {
      try {
        return Model.fromJson(JSON.parse(saved))
      } catch {
        /* ignore */
      }
    }
    return Model.fromJson(defaultLayout)
  })

  const factory = useCallback((node: TabNode) => {
    const id = node.getComponent()
    const Widget = getWidget(id)
    if (Widget) return <Widget />
    return <div>Unknown widget: {id}</div>
  }, [])

  const handleModelChange = useCallback(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(model.toJson()))
  }, [model])

  if (process.env.NODE_ENV === 'test') {
    return (
      <div>
        {listWidgets().map((key) => {
          const Widget = getWidget(key)
          return Widget ? <Widget key={key} /> : null
        })}
      </div>
    )
  }

  return <Layout model={model} factory={factory} onModelChange={handleModelChange} />
}
