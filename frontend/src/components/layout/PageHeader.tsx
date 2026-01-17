import { ReactNode } from 'react'

interface PageHeaderProps {
    title: string
    subtitle?: string
    actions?: ReactNode
}

export function PageHeader({ title, subtitle, actions }: PageHeaderProps) {
    return (
        <div className="page-header">
            <div className="page-header-content">
                <h1 className="page-title">{title}</h1>
                {subtitle && <p className="page-subtitle">{subtitle}</p>}
            </div>
            {actions && <div className="page-actions">{actions}</div>}

            <style>{`
                .page-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: flex-start;
                    margin-bottom: var(--space-6);
                }

                .page-header-content {
                    flex: 1;
                }

                .page-title {
                    font-size: var(--text-2xl);
                    font-weight: var(--font-bold);
                    color: var(--text-primary);
                    margin: 0;
                    line-height: var(--leading-tight);
                }

                .page-subtitle {
                    margin-top: var(--space-1);
                    font-size: var(--text-sm);
                    color: var(--text-secondary);
                }

                .page-actions {
                    display: flex;
                    align-items: center;
                    gap: var(--space-2);
                }

                @media (max-width: 640px) {
                    .page-header {
                        flex-direction: column;
                        gap: var(--space-4);
                    }

                    .page-actions {
                        width: 100%;
                        justify-content: flex-start;
                    }
                }
            `}</style>
        </div>
    )
}
