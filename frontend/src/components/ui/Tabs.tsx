import { useState } from 'react'
import clsx from 'clsx'

interface Tab {
    id: string;
    label: string;
    icon?: React.ReactNode;
}

interface TabsProps {
    tabs: Tab[];
    activeTab: string;
    onChange: (tabId: string) => void;
}

export function Tabs({ tabs, activeTab, onChange }: TabsProps) {
    return (
        <div className="tabs">
            {tabs.map(tab => (
                <button
                    key={tab.id}
                    className={clsx('tab', activeTab === tab.id && 'active')}
                    onClick={() => onChange(tab.id)}
                >
                    {tab.icon}
                    {tab.label}
                </button>
            ))}
        </div>
    )
}

// Controlled tabs with panel content
interface TabPanelProps {
    children: React.ReactNode;
    value: string;
    activeValue: string;
}

export function TabPanel({ children, value, activeValue }: TabPanelProps) {
    if (value !== activeValue) return null;
    return <div className="tab-panel">{children}</div>;
}
