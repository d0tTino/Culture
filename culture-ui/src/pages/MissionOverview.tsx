import { useState } from 'react'
import clsx from 'clsx'
import {
  ColumnDef,
  Row,
  flexRender,
  getCoreRowModel,
  useReactTable,
} from '@tanstack/react-table'
import missionsData from '../mock/missions.json'
import {
  DndContext,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors,
} from '@dnd-kit/core'
import {
  arrayMove,
  SortableContext,
  sortableKeyboardCoordinates,
  useSortable,
  verticalListSortingStrategy,
} from '@dnd-kit/sortable'
import { CSS } from '@dnd-kit/utilities'

interface Mission {
  id: number
  name: string
  status: string
  progress: number
}

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
  const [data, setData] = useState<Mission[]>(missionsData as Mission[])

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

  return (
    <div className="p-4">
      <h2 className="mb-4 text-xl font-bold">Mission Overview</h2>
      <DndContext
        sensors={sensors}
        onDragEnd={({ active, over }) => {
          if (over && active.id !== over.id) {
            const oldIndex = table.getRowModel().rows.findIndex((r) => r.id === active.id)
            const newIndex = table.getRowModel().rows.findIndex((r) => r.id === over.id)
            setData((items) => arrayMove(items, oldIndex, newIndex))
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
