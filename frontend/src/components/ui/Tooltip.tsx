import { ReactNode, useState } from 'react'

interface TooltipProps {
    content: ReactNode
    children: ReactNode
    position?: 'top' | 'bottom' | 'left' | 'right'
}

export function Tooltip({ content, children, position = 'top' }: TooltipProps) {
    const [isVisible, setIsVisible] = useState(false)

    const positionStyles: Record<string, React.CSSProperties> = {
        top: {
            bottom: '100%',
            left: '50%',
            transform: 'translateX(-50%)',
            marginBottom: '8px',
        },
        bottom: {
            top: '100%',
            left: '50%',
            transform: 'translateX(-50%)',
            marginTop: '8px',
        },
        left: {
            right: '100%',
            top: '50%',
            transform: 'translateY(-50%)',
            marginRight: '8px',
        },
        right: {
            left: '100%',
            top: '50%',
            transform: 'translateY(-50%)',
            marginLeft: '8px',
        },
    }

    return (
        <div
            style={{ position: 'relative', display: 'inline-flex' }}
            onMouseEnter={() => setIsVisible(true)}
            onMouseLeave={() => setIsVisible(false)}
        >
            {children}
            {isVisible && (
                <div
                    style={{
                        position: 'absolute',
                        zIndex: 50,
                        padding: 'var(--space-2) var(--space-3)',
                        fontSize: 'var(--text-xs)',
                        fontWeight: 'var(--font-medium)',
                        color: 'var(--text-primary)',
                        backgroundColor: 'var(--bg-elevated)',
                        border: '1px solid var(--border-default)',
                        borderRadius: 'var(--radius-md)',
                        boxShadow: 'var(--shadow-lg)',
                        whiteSpace: 'nowrap',
                        pointerEvents: 'none',
                        animation: 'fade-in var(--transition-fast)',
                        ...positionStyles[position],
                    }}
                >
                    {content}
                </div>
            )}
        </div>
    )
}
