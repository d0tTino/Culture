export type WidgetComponent = React.ComponentType<unknown>;

const registry: Record<string, WidgetComponent> = {};

export function registerWidget(id: string, component: WidgetComponent): void {
  registry[id] = component;
}

export function getWidget(id: string): WidgetComponent | undefined {
  return registry[id];
}

export function listWidgets(): [string, WidgetComponent][] {
  return Object.entries(registry);
}
