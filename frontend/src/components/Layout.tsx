import { Link, useLocation } from 'react-router-dom';
import { clsx } from 'clsx';

interface LayoutProps {
  children: React.ReactNode;
}

const navItems = [
  { path: '/', label: 'Markets' },
  { path: '/props', label: 'Props' },
  { path: '/trends', label: 'Trends' },
  { path: '/evaluation', label: 'Evaluation' },
];

export function Layout({ children }: LayoutProps) {
  const location = useLocation();

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            {/* Logo */}
            <Link to="/" className="flex items-center space-x-2">
              <span className="text-xl font-bold text-gray-900">NBA Value</span>
              <span className="text-sm text-gray-500">Beta</span>
            </Link>

            {/* Navigation */}
            <nav className="flex space-x-8">
              {navItems.map((item) => (
                <Link
                  key={item.path}
                  to={item.path}
                  className={clsx(
                    'text-sm font-medium transition-colors',
                    location.pathname === item.path
                      ? 'text-blue-600'
                      : 'text-gray-500 hover:text-gray-900'
                  )}
                >
                  {item.label}
                </Link>
              ))}
            </nav>

            {/* User menu placeholder */}
            <div className="flex items-center space-x-4">
              <button className="btn-primary text-sm">Sign In</button>
            </div>
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">{children}</main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-auto">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <p className="text-sm text-gray-500 text-center">
            NBA Value Betting Platform - For entertainment purposes only
          </p>
        </div>
      </footer>
    </div>
  );
}
