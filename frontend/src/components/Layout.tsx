import { Link, useLocation } from 'react-router-dom';
import { clsx } from 'clsx';

interface LayoutProps {
  children: React.ReactNode;
}

// Nav is two levels: pick a sport, then a page within it. Previously all nine
// links sat in one flat row, which put NBA's own sub-pages (Props / Trends /
// Performance) at the same level as the sports themselves.
const SPORTS = [
  {
    key: 'nba',
    label: 'NBA',
    icon: '🏀',
    path: '/',
    pages: [
      { path: '/', label: 'Picks' },
      { path: '/props', label: 'Props' },
      { path: '/trends', label: 'Trends' },
      { path: '/evaluation', label: 'Results' },
    ],
  },
  {
    key: 'mlb',
    label: 'MLB',
    icon: '⚾',
    path: '/mlb',
    pages: [
      { path: '/mlb', label: 'Picks' },
      { path: '/nrfi', label: 'NRFI' },
      { path: '/mlb/performance', label: 'Results' },
    ],
  },
  {
    key: 'nfl',
    label: 'NFL',
    icon: '🏈',
    path: '/nfl',
    pages: [
      { path: '/nfl', label: 'Picks' },
      { path: '/nfl/performance', label: 'Results' },
    ],
  },
];

/** Which sport owns the current route. Total — everything else is NBA,
 *  including '/', '/props', '/trends', '/evaluation' and '/bet/:marketId'. */
function activeSport(pathname: string) {
  if (pathname.startsWith('/nfl')) return SPORTS[2];
  if (pathname.startsWith('/mlb') || pathname === '/nrfi') return SPORTS[1];
  return SPORTS[0];
}

export function Layout({ children }: LayoutProps) {
  const location = useLocation();
  const current = activeSport(location.pathname);

  return (
    <div className="min-h-screen bg-tru-bg bg-grid">
      {/* Header */}
      <header className="border-b border-tru-border bg-tru-bg/80 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-14">
            {/* Logo */}
            <Link to="/" className="flex items-center gap-1.5 group">
              <img
                src="/favicon.png"
                alt=""
                className="w-7 h-7 group-hover:brightness-125 transition-all"
              />
              <span className="text-xl font-display font-bold tracking-tight">
                <span className="bg-gradient-to-r from-sky-400 to-accent-cyan bg-clip-text text-transparent">Tru</span><span className="text-white">Line</span>
              </span>
            </Link>

            {/* Sport switcher — three tabs, all screen sizes */}
            <nav className="flex items-center gap-1">
              {SPORTS.map((sport) => {
                const isActive = sport.key === current.key;
                return (
                  <Link
                    key={sport.key}
                    to={sport.path}
                    className={clsx(
                      'px-2.5 sm:px-3 py-1.5 rounded-lg text-sm font-medium transition-all',
                      isActive
                        ? 'bg-accent-cyan/10 text-accent-cyan'
                        : 'text-txt-muted hover:text-txt-primary hover:bg-tru-surface'
                    )}
                  >
                    <span className="mr-1">{sport.icon}</span>
                    {sport.label}
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
                className="hidden sm:block text-txt-muted hover:text-txt-primary transition-colors text-sm"
              >
                @trulineapp
              </a>
            </div>
          </div>
        </div>
      </header>

      {/* Pages within the active sport. Hidden when there's only one page. */}
      {current.pages.length > 1 && (
        <nav className="flex overflow-x-auto border-b border-tru-border bg-tru-surface/50 gap-1 py-2 px-4 sm:px-6 lg:px-8">
          <div className="max-w-7xl w-full mx-auto flex gap-1">
            {current.pages.map((page) => {
              const isActive = location.pathname === page.path;
              return (
                <Link
                  key={page.path}
                  to={page.path}
                  className={clsx(
                    'flex-shrink-0 px-3 py-1.5 rounded-lg text-sm font-medium transition-all',
                    isActive
                      ? 'bg-accent-cyan/10 text-accent-cyan'
                      : 'text-txt-muted hover:text-txt-primary hover:bg-tru-surface'
                  )}
                >
                  {page.label}
                </Link>
              );
            })}
          </div>
        </nav>
      )}

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
