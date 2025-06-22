import { useState, useEffect } from 'react'
import clsx from 'clsx'
import type { ColumnDef, Row } from '@tanstack/react-table'
import {
  flexRender,
  getCoreRowModel,
  useReactTable,
} from '@tanstack/react-table'
import { fetchMissions, type Mission } from '../lib/api'
import {
  DndContext,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors,
} from '@dnd-kit/core'
import {
  SortableContext,
  sortableKeyboardCoordinates,
  useSortable,
  verticalListSortingStrategy,
} from '@dnd-kit/sortable'
import { reorderMissions } from '../lib/reorderMissions'
import { CSS } from '@dnd-kit/utilities'

function DraggableRow({ row }: { row: Row<Mission> }) {
  const { attributes, listeners, setNodeRef, transform, transition } = useSortable({
    id: row.id,
  })

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
  }

  return (
    <tr
      ref={setNodeRef}
      style={style}
      {...attributes}
      {...listeners}
      className={clsx('border-b', 'hover:bg-muted/50')}
    >
      {row.getVisibleCells().map((cell) => (
        <td key={cell.id} className="p-2">
          {flexRender(cell.column.columnDef.cell, cell.getContext())}
        </td>
      ))}
    </tr>
  )
}

export default function MissionOverview() {
  const [data, setData] = useState<Mission[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<Error | null>(null)

  useEffect(() => {
    fetchMissions()
      .then(setData)
      .catch((err) => setError(err as Error))
      .finally(() => setLoading(false))
  }, [])

  const columns: ColumnDef<Mission>[] = [
    {
      accessorKey: 'id',
      header: 'ID',
    },
    {
      accessorKey: 'name',
      header: 'Name',
    },
    {
      accessorKey: 'status',
      header: 'Status',
    },
    {
      accessorKey: 'progress',
      header: 'Progress',
      cell: (info) => `${info.getValue()}%`,
    },
  ]

  const table = useReactTable({
    data,
    columns,
    getRowId: (row) => row.id.toString(),
    getCoreRowModel: getCoreRowModel(),
  })

  const sensors = useSensors(
    useSensor(PointerSensor),
    useSensor(KeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
    })
  )

  if (loading) {
    return <div className="p-4">Loading missions...</div>
  }

  if (error) {
    return <div className="p-4 text-red-500">Error loading missions</div>
  }

  return (
    <div className="p-4">
      <h2 className="mb-4 text-xl font-bold">Mission Overview</h2>
      <DndContext
        sensors={sensors}
        onDragEnd={({ active, over }) => {
          if (over && active.id !== over.id) {
            setData((items) => {
              const from = items.findIndex((m) => m.id === Number(active.id))
              const to = items.findIndex((m) => m.id === Number(over.id))
              if (from === -1 || to === -1) {
                return items
              }
              return reorderMissions(items, from, to)
            })
          }
        }}
      >
        <SortableContext items={table.getRowModel().rows} strategy={verticalListSortingStrategy}>
          <table className="w-full border">
            <thead>
              {table.getHeaderGroups().map((headerGroup) => (
                <tr key={headerGroup.id} className="bg-muted/50">
                  {headerGroup.headers.map((header) => (
                    <th key={header.id} className="p-2 text-left">
                      {flexRender(header.column.columnDef.header, header.getContext())}
                    </th>
                  ))}
                </tr>
              ))}
            </thead>
            <tbody>
              {table.getRowModel().rows.map((row) => (
                <DraggableRow key={row.id} row={row} />
              ))}
            </tbody>
          </table>
        </SortableContext>
      </DndContext>
    </div>
  )

}
