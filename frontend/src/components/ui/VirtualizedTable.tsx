// =============================================================================
// Virtualized Table - High-performance table for large datasets
// =============================================================================

import { useRef, ReactNode } from 'react';
import { useVirtualizer } from '@tanstack/react-virtual';

export interface TableColumn<T> {
    key: string;
    header: string;
    render: (row: T) => ReactNode;
    width?: string;
    align?: 'left' | 'center' | 'right';
}

interface VirtualizedTableProps<T> {
    data: T[];
    columns: TableColumn<T>[];
    rowHeight?: number;
    maxHeight?: number;
    onRowClick?: (row: T) => void;
    getRowKey: (row: T) => string | number;
}

export function VirtualizedTable<T>({
    data,
    columns,
    rowHeight = 56,
    maxHeight = 600,
    onRowClick,
    getRowKey,
}: VirtualizedTableProps<T>) {
    const parentRef = useRef<HTMLDivElement>(null);

    const virtualizer = useVirtualizer({
        count: data.length,
        getScrollElement: () => parentRef.current,
        estimateSize: () => rowHeight,
        overscan: 5,
    });

    const virtualItems = virtualizer.getVirtualItems();

    return (
        <div className="virtualized-table-container">
            {/* Header */}
            <div className="table-header">
                {columns.map((column) => (
                    <div
                        key={column.key}
                        className="table-header-cell"
                        style={{
                            width: column.width,
                            textAlign: column.align || 'left',
                        }}
                    >
                        {column.header}
                    </div>
                ))}
            </div>

            {/* Scrollable body */}
            <div
                ref={parentRef}
                className="table-body"
                style={{ maxHeight }}
            >
                {data.length === 0 ? (
                    <div className="table-empty">No data available</div>
                ) : (
                    <div
                        className="virtual-rows"
                        style={{
                            height: `${virtualizer.getTotalSize()}px`,
                            position: 'relative',
                        }}
                    >
                        {virtualItems.map((virtualRow) => {
                            const row = data[virtualRow.index];
                            const rowKey = getRowKey(row);

                            return (
                                <div
                                    key={rowKey}
                                    className={`table-row ${onRowClick ? 'clickable' : ''}`}
                                    style={{
                                        position: 'absolute',
                                        top: 0,
                                        left: 0,
                                        width: '100%',
                                        height: `${virtualRow.size}px`,
                                        transform: `translateY(${virtualRow.start}px)`,
                                    }}
                                    onClick={() => onRowClick?.(row)}
                                >
                                    {columns.map((column) => (
                                        <div
                                            key={column.key}
                                            className="table-cell"
                                            style={{
                                                width: column.width,
                                                textAlign: column.align || 'left',
                                            }}
                                        >
                                            {column.render(row)}
                                        </div>
                                    ))}
                                </div>
                            );
                        })}
                    </div>
                )}
            </div>

            {/* Row count footer */}
            {data.length > 0 && (
                <div className="table-footer">
                    Showing {virtualItems.length} of {data.length} rows
                </div>
            )}

            <style>{`
                .virtualized-table-container {
                    background: var(--bg-surface);
                    border: 1px solid var(--border-subtle);
                    border-radius: var(--radius-lg);
                    overflow: hidden;
                }

                .table-header {
                    display: flex;
                    align-items: center;
                    padding: var(--space-3) var(--space-4);
                    background: var(--bg-elevated);
                    border-bottom: 1px solid var(--border-subtle);
                    position: sticky;
                    top: 0;
                    z-index: 1;
                }

                .table-header-cell {
                    flex: 1;
                    font-size: var(--text-sm);
                    font-weight: var(--font-semibold);
                    color: var(--text-secondary);
                    text-transform: uppercase;
                    letter-spacing: 0.05em;
                }

                .table-body {
                    overflow-y: auto;
                }

                .virtual-rows {
                    width: 100%;
                }

                .table-row {
                    display: flex;
                    align-items: center;
                    padding: 0 var(--space-4);
                    border-bottom: 1px solid var(--border-subtle);
                    transition: background-color 0.15s;
                }

                .table-row.clickable {
                    cursor: pointer;
                }

                .table-row.clickable:hover {
                    background: var(--bg-hover);
                }

                .table-cell {
                    flex: 1;
                    font-size: var(--text-sm);
                    color: var(--text-primary);
                    overflow: hidden;
                    text-overflow: ellipsis;
                    white-space: nowrap;
                }

                .table-empty {
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    height: 200px;
                    color: var(--text-secondary);
                }

                .table-footer {
                    padding: var(--space-2) var(--space-4);
                    font-size: var(--text-xs);
                    color: var(--text-tertiary);
                    background: var(--bg-elevated);
                    border-top: 1px solid var(--border-subtle);
                }
            `}</style>
        </div>
    );
}
