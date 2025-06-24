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
export const registerWidget = widgetRegistry.register.bind(widgetRegistry)
export const getWidget = widgetRegistry.get.bind(widgetRegistry)
export const listWidgets = widgetRegistry.list.bind(widgetRegistry)


