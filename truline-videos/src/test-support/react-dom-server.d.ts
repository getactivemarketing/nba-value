/**
 * Minimal ambient types for `react-dom/server`.
 *
 * @types/react-dom is not a dependency of this package and adding it would
 * mean regenerating the lockfile, which the exact Remotion/React pins exist to
 * avoid churn on (see the toPublicSrc comment in
 * scripts/render-pick-previews.ts). Only the one function the composition
 * tests use is declared here — react-dom itself is already installed.
 */
declare module 'react-dom/server' {
  import type { ReactElement } from 'react';
  export function renderToStaticMarkup(element: ReactElement): string;
  export function renderToString(element: ReactElement): string;
}
