# Culture UI Widget System

Widgets are React components registered in a simple registry. The default UI presents them on pages navigated via a sidebar and `react-router-dom` routes. Applications may optionally use the `DockManager` to provide a draggable layout.

```ts
import { registerWidget } from '@/lib/widgetRegistry'
import MyWidget from './MyWidget'

registerWidget('myWidget', MyWidget)
```

When the optional `DockManager` is enabled it persists the current layout to `localStorage` under the key `dockLayout` and restores this layout on the next load.
