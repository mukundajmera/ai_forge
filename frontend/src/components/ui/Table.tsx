import { ReactNode } from 'react'
import { Skeleton } from './Skeleton'

interface Column<T> {
    key: string
    header: string
    render: (row: T) => ReactNode
    width?: string
}

interface TableProps<T> {
    columns: Column<T>[]
    data: T[]
    loading?: boolean
    emptyMessage?: string
    onRowClick?: (row: T) => void
}

export function Table<T extends { id: string | number }>({
    columns,
    data,
    loading,
    emptyMessage = 'No data available',
    onRowClick,
}: TableProps<T>) {
    if (loading) {
        return (
            <div style={{ padding: 'var(--space-4)' }}>
                {[...Array(5)].map((_, i) => (
                    <Skeleton key={i} height={48} style={{ marginBottom: 'var(--space-2)' }} />
                ))}
            </div>
        )
    }

    if (!data.length) {
        return (
            <div style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                padding: 'var(--space-12)',
            }}>
                <p style={{
                    fontSize: 'var(--text-sm)',
                    color: 'var(--text-secondary)',
                }}>
                    {emptyMessage}
                </p>
            </div>
        )
    }

    return (
        <div className="table-container">
            <table className="table">
                <thead>
                    <tr>
                        {columns.map((column) => (
                            <th
                                key={column.key}
                                style={{ width: column.width }}
                            >
                                {column.header}
                            </th>
                        ))}
                    </tr>
                </thead>
                <tbody>
                    {data.map((row) => (
                        <tr
                            key={row.id}
                            onClick={() => onRowClick?.(row)}
                            style={{
                                cursor: onRowClick ? 'pointer' : undefined,
                            }}
                        >
                            {columns.map((column) => (
                                <td key={column.key}>
                                    {column.render(row)}
                                </td>
                            ))}
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    )
}
