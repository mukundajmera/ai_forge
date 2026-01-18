import { useState } from 'react'
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
  HelpCircle,
  Menu,
  X
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
  { to: '/jobs', label: 'Jobs', icon: Activity },
  { to: '/models', label: 'Models', icon: Boxes },
  { to: '/monitoring', label: 'Monitoring', icon: BarChart3 },
  { to: '/missions', label: 'Missions', icon: Sparkles },
]

// Bottom nav items for mobile (subset of main nav)
const bottomNavItems = mainNavItems.slice(0, 4)

export function PageLayout() {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false)

  const toggleMobileMenu = () => setIsMobileMenuOpen(!isMobileMenuOpen)
  const closeMobileMenu = () => setIsMobileMenuOpen(false)

  return (
    <div className="page-layout">
      {/* Mobile Header */}
      <header className="mobile-header">
        <div className="logo">
          <Zap className="logo-icon" />
          <span className="logo-text">AI Forge</span>
        </div>
        <button
          className="hamburger-btn"
          onClick={toggleMobileMenu}
          aria-label="Toggle menu"
        >
          {isMobileMenuOpen ? <X size={24} /> : <Menu size={24} />}
        </button>
      </header>

      {/* Mobile Overlay */}
      {isMobileMenuOpen && (
        <div className="mobile-overlay" onClick={closeMobileMenu} />
      )}

      {/* Sidebar - visible on desktop, slide-in on mobile */}
      <aside className={`sidebar ${isMobileMenuOpen ? 'open' : ''}`}>
        <div className="sidebar-header">
          <div className="logo">
            <Zap className="logo-icon" />
            <div className="logo-content">
              <span className="logo-text">AI Forge</span>
              <span className="logo-env">Local Mac</span>
            </div>
          </div>
          <button
            className="close-sidebar-btn"
            onClick={closeMobileMenu}
            aria-label="Close menu"
          >
            <X size={20} />
          </button>
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
              onClick={closeMobileMenu}
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
          <NavLink to="/settings" className="nav-item" onClick={closeMobileMenu}>
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

      {/* Mobile Bottom Navigation */}
      <nav className="bottom-nav">
        {bottomNavItems.map(({ to, label, icon: Icon, badge }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              `bottom-nav-item ${isActive ? 'active' : ''}`
            }
          >
            <Icon size={20} />
            <span>{label}</span>
            {badge && badge > 0 && (
              <span className="bottom-nav-badge">{badge}</span>
            )}
          </NavLink>
        ))}
      </nav>

      <style>{`
        .page-layout {
          display: flex;
          min-height: 100vh;
        }
        
        /* Mobile Header - hidden on desktop */
        .mobile-header {
          display: none;
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          height: 56px;
          background: var(--bg-surface);
          border-bottom: 1px solid var(--border);
          padding: 0 var(--space-4);
          align-items: center;
          justify-content: space-between;
          z-index: 50;
        }
        
        .mobile-header .logo {
          display: flex;
          align-items: center;
          gap: var(--space-2);
        }
        
        .mobile-header .logo-text {
          font-size: var(--text-lg);
          font-weight: 700;
          background: linear-gradient(135deg, var(--primary-400), var(--primary-300));
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
        }
        
        .hamburger-btn {
          background: none;
          border: none;
          color: var(--text-primary);
          cursor: pointer;
          padding: var(--space-2);
          border-radius: var(--radius-md);
          display: flex;
          align-items: center;
          justify-content: center;
        }
        
        .hamburger-btn:hover {
          background: var(--bg-hover);
        }
        
        /* Mobile Overlay */
        .mobile-overlay {
          display: none;
          position: fixed;
          inset: 0;
          background: rgba(0, 0, 0, 0.5);
          z-index: 45;
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
        
        .close-sidebar-btn {
          display: none;
          background: none;
          border: none;
          color: var(--text-secondary);
          cursor: pointer;
          padding: var(--space-2);
          border-radius: var(--radius-md);
        }
        
        .close-sidebar-btn:hover {
          background: var(--bg-hover);
          color: var(--text-primary);
        }
        
        .sidebar-header {
          padding: var(--space-4);
          border-bottom: 1px solid var(--border);
          display: flex;
          align-items: center;
          justify-content: space-between;
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
          overflow-y: auto;
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
        
        /* Bottom Navigation - hidden on desktop */
        .bottom-nav {
          display: none;
          position: fixed;
          bottom: 0;
          left: 0;
          right: 0;
          height: 64px;
          background: var(--bg-surface);
          border-top: 1px solid var(--border);
          z-index: 50;
        }
        
        .bottom-nav-item {
          flex: 1;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          gap: 2px;
          color: var(--text-tertiary);
          text-decoration: none;
          font-size: 10px;
          font-weight: 500;
          transition: color var(--transition-fast);
          position: relative;
        }
        
        .bottom-nav-item:hover,
        .bottom-nav-item.active {
          color: var(--primary-400);
        }
        
        .bottom-nav-badge {
          position: absolute;
          top: 4px;
          right: calc(50% - 16px);
          min-width: 16px;
          height: 16px;
          padding: 0 4px;
          background: var(--primary-500);
          color: white;
          font-size: 9px;
          font-weight: 600;
          border-radius: var(--radius-full);
          display: flex;
          align-items: center;
          justify-content: center;
        }
        
        /* ================================
           TABLET BREAKPOINT (768px)
           ================================ */
        @media (max-width: 768px) {
          .mobile-header {
            display: flex;
          }
          
          .mobile-overlay {
            display: block;
          }
          
          .sidebar {
            transform: translateX(-100%);
            transition: transform var(--transition-base);
            z-index: 50;
          }
          
          .sidebar.open {
            transform: translateX(0);
          }
          
          .close-sidebar-btn {
            display: flex;
          }
          
          .main-content {
            margin-left: 0;
            padding-top: 56px;
            padding-bottom: 64px;
          }
          
          .bottom-nav {
            display: flex;
          }
        }
        
        /* ================================
           MOBILE BREAKPOINT (480px)
           ================================ */
        @media (max-width: 480px) {
          .sidebar {
            width: 100%;
          }
        }
      `}</style>
    </div>
  )
}

