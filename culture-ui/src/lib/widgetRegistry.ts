export interface WidgetRegistry {
  register(name: string, component: React.ComponentType): void
  get(name: string): React.ComponentType | undefined
  list(): string[]
}

class Registry implements WidgetRegistry {
  private widgets = new Map<string, React.ComponentType>()

  register(name: string, component: React.ComponentType) {
    this.widgets.set(name, component)
  }

  get(name: string) {
    return this.widgets.get(name)
  }

  list() {
    return Array.from(this.widgets.keys())
  }
}

export const widgetRegistry = new Registry()

export function registerWidget(name: string, component: React.ComponentType) {
  widgetRegistry.register(name, component)
}

export function getWidget(name: string) {
  return widgetRegistry.get(name)
}

export function listWidgets() {
  return widgetRegistry.list()
}

