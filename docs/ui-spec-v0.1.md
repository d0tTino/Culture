# Culture UI Widget System

Widgets are React components that can be docked using the `DockManager`.
New widgets must be registered during application startup so the layout can
load them by identifier. The registry exposes `registerWidget`, `getWidget`,
and `listWidgets` helpers.

```
import { registerWidget, getWidget } from '@/lib/widgetRegistry'
import MyWidget from './MyWidget'

registerWidget('myWidget', MyWidget)
// later
const Widget = getWidget('myWidget')
```

Tests can use `listWidgets()` to access all registered components.

The `DockManager` persists the current layout to `localStorage` under the key
`dockLayout`. When the application loads, it restores this layout or falls back
to the provided default layout.
