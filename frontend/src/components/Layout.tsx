import { Link, useLocation } from 'react-router-dom';
import { clsx } from 'clsx';

interface LayoutProps {
  children: React.ReactNode;
}

const navItems = [
  { path: '/', label: 'NBA', icon: '🏀' },
  { path: '/mlb', label: 'MLB', icon: '⚾' },
  { path: '/props', label: 'Props' },
  { path: '/trends', label: 'Trends' },
  { path: '/evaluation', label: 'Performance' },
];

export function Layout({ children }: LayoutProps) {
  const location = useLocation();

  return (
    <div className="min-h-screen bg-tru-bg bg-grid">
      {/* Header */}
      <header className="border-b border-tru-border bg-tru-bg/80 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-14">
            {/* Logo */}
            <Link to="/" className="flex items-center gap-2 group">
              <img
                src="/favicon.png"
                alt="TruLine"
                className="w-8 h-8 group-hover:brightness-125 transition-all"
              />
              <img
                src="/logo-dark.png"
                alt="TruLine"
                className="h-7 hidden sm:block"
              />
            </Link>

            {/* Navigation */}
            <nav className="hidden md:flex items-center gap-1">
              {navItems.map((item) => {
                const isActive = location.pathname === item.path;
                return (
                  <Link
                    key={item.path}
                    to={item.path}
                    className={clsx(
                      'px-3 py-1.5 rounded-lg text-sm font-medium transition-all',
                      isActive
                        ? 'bg-accent-cyan/10 text-accent-cyan'
                        : 'text-txt-muted hover:text-txt-primary hover:bg-tru-surface'
                    )}
                  >
                    {item.icon && <span className="mr-1">{item.icon}</span>}
                    {item.label}
                  </Link>
                );
              })}
            </nav>

            {/* Right side */}
            <div className="flex items-center gap-3">
              <a
                href="https://x.com/trulineapp"
                target="_blank"
                rel="noopener noreferrer"
                className="text-txt-muted hover:text-txt-primary transition-colors text-sm"
              >
                @trulineapp
              </a>
            </div>
          </div>
        </div>
      </header>

      {/* Mobile nav */}
      <nav className="md:hidden flex overflow-x-auto border-b border-tru-border bg-tru-surface/50 px-4 gap-1 py-2">
        {navItems.map((item) => {
          const isActive = location.pathname === item.path;
          return (
            <Link
              key={item.path}
              to={item.path}
              className={clsx(
                'flex-shrink-0 px-3 py-1.5 rounded-lg text-sm font-medium transition-all',
                isActive
                  ? 'bg-accent-cyan/10 text-accent-cyan'
                  : 'text-txt-muted'
              )}
            >
              {item.label}
            </Link>
          );
        })}
      </nav>

      {/* Main content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {children}
      </main>

      {/* Footer */}
      <footer className="border-t border-tru-border mt-auto">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex justify-between items-center">
            <p className="text-xs text-txt-muted">
              TruLine - AI Sports Betting Intelligence. For entertainment purposes only.
            </p>
            <p className="text-xs text-txt-muted font-mono">
              truline.app
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}
