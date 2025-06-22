# Culture UI Widget System

Widgets are React components that can be docked using the `DockManager`.
New widgets must be registered during application startup so the layout can
load them by identifier.

```
import { registerWidget } from '@/lib/widgetRegistry'
import MyWidget from './MyWidget'

registerWidget('myWidget', MyWidget)
```

The `DockManager` persists the current layout to `localStorage` under the key
`dockLayout`. When the application loads, it restores this layout or falls back
to the provided default layout.
