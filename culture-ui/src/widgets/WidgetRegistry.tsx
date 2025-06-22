import type { ComponentType } from 'react'

export type WidgetComponent = ComponentType<unknown>

const registry = new Map<string, WidgetComponent>()

export function registerWidget(name: string, component: WidgetComponent) {
  registry.set(name, component)
}

export function getWidget(name: string): WidgetComponent | undefined {
  return registry.get(name)
}

export function listWidgets(): string[] {
  return Array.from(registry.keys())
}
