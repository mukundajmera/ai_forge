import { Outlet, NavLink } from 'react-router-dom'
import {
  LayoutDashboard,
  FolderOpen,
  Activity,
  Boxes,
  BarChart3,
  Sparkles,
  Settings,
  Zap,
  HelpCircle
} from 'lucide-react'

interface NavItem {
  to: string;
  label: string;
  icon: typeof LayoutDashboard;
  badge?: number;
}

const mainNavItems: NavItem[] = [
  { to: '/', label: 'Dashboard', icon: LayoutDashboard },
  { to: '/datasets', label: 'Datasets', icon: FolderOpen },
  { to: '/jobs', label: 'Jobs', icon: Activity, badge: 2 },
  { to: '/models', label: 'Models', icon: Boxes },
  { to: '/monitoring', label: 'Monitoring', icon: BarChart3 },
  { to: '/missions', label: 'Missions', icon: Sparkles, badge: 1 },
]

export function PageLayout() {
  return (
    <div className="page-layout">
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="logo">
            <Zap className="logo-icon" />
            <div className="logo-content">
              <span className="logo-text">AI Forge</span>
              <span className="logo-env">Local Mac</span>
            </div>
          </div>
        </div>

        <nav className="sidebar-nav">
          {mainNavItems.map(({ to, label, icon: Icon, badge }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                `nav-item ${isActive ? 'active' : ''}`
              }
            >
              <Icon size={18} />
              <span>{label}</span>
              {badge && badge > 0 && (
                <span className="nav-badge">{badge}</span>
              )}
            </NavLink>
          ))}
        </nav>

        <div className="sidebar-footer">
          <NavLink to="/settings" className="nav-item">
            <Settings size={18} />
            <span>Settings</span>
          </NavLink>
          <button className="nav-item">
            <HelpCircle size={18} />
            <span>Help</span>
          </button>
        </div>
      </aside>

      <main className="main-content">
        <Outlet />
      </main>

      <style>{`
        .page-layout {
          display: flex;
          min-height: 100vh;
        }
        
        .sidebar {
          width: 240px;
          background-color: var(--bg-surface);
          border-right: 1px solid var(--border);
          display: flex;
          flex-direction: column;
          position: fixed;
          top: 0;
          left: 0;
          bottom: 0;
          z-index: 40;
        }
        
        .sidebar-header {
          padding: var(--space-4);
          border-bottom: 1px solid var(--border);
        }
        
        .logo {
          display: flex;
          align-items: center;
          gap: var(--space-3);
        }
        
        .logo-icon {
          color: var(--primary-400);
          width: 28px;
          height: 28px;
        }
        
        .logo-content {
          display: flex;
          flex-direction: column;
        }
        
        .logo-text {
          font-size: var(--text-lg);
          font-weight: 700;
          background: linear-gradient(135deg, var(--primary-400), var(--primary-300));
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
          line-height: 1.2;
        }
        
        .logo-env {
          font-size: var(--text-xs);
          color: var(--text-muted);
        }
        
        .sidebar-nav {
          flex: 1;
          padding: var(--space-4);
          display: flex;
          flex-direction: column;
          gap: var(--space-1);
        }
        
        .nav-item {
          display: flex;
          align-items: center;
          gap: var(--space-3);
          padding: var(--space-3) var(--space-4);
          color: var(--text-secondary);
          border-radius: var(--radius-md);
          font-size: var(--text-sm);
          font-weight: 500;
          text-decoration: none;
          transition: all var(--transition-fast);
          border: none;
          background: none;
          cursor: pointer;
          width: 100%;
          text-align: left;
        }
        
        .nav-item:hover {
          background-color: var(--bg-hover);
          color: var(--text-primary);
        }
        
        .nav-item.active {
          background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(99, 102, 241, 0.05));
          color: var(--primary-400);
        }
        
        .nav-badge {
          margin-left: auto;
          padding: 0 6px;
          min-width: 18px;
          height: 18px;
          display: flex;
          align-items: center;
          justify-content: center;
          background: var(--primary-500);
          color: white;
          font-size: var(--text-xs);
          font-weight: 600;
          border-radius: var(--radius-full);
        }
        
        .sidebar-footer {
          padding: var(--space-4);
          border-top: 1px solid var(--border);
          display: flex;
          flex-direction: column;
          gap: var(--space-1);
        }
        
        .main-content {
          flex: 1;
          margin-left: 240px;
          min-height: 100vh;
          background-color: var(--bg-base);
        }
        
        @media (max-width: 768px) {
          .sidebar {
            transform: translateX(-100%);
          }
          
          .main-content {
            margin-left: 0;
          }
        }
      `}</style>
    </div>
  )
}
