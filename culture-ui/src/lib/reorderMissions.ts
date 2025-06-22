import { arrayMove } from "@dnd-kit/sortable"
export function reorderMissions<T extends { id: number }>(data: T[], activeId: number, overId: number) {
  const oldIndex = data.findIndex((r) => r.id === activeId);
  const newIndex = data.findIndex((r) => r.id === overId);
  return arrayMove(data, oldIndex, newIndex);
}
