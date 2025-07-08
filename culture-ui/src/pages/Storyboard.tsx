import Storyboard from '../widgets/Storyboard'
import { registerWidget } from '../lib/widgetRegistry'

export default function StoryboardPage() {
  return (
    <div className="p-4 space-y-4">
      <h1 className="text-xl font-bold">Storyboard</h1>
      <Storyboard />
    </div>
  )
}

registerWidget('Storyboard', StoryboardPage)
