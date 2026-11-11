// @vitest-environment jsdom
import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';
import { FormCell } from './FormCell';

afterEach(cleanup);

describe('FormCell', () => {
  it('shows hits of games and the market chance in the title', () => {
    render(<FormCell form={{ hits: 7, games: 10, rate: 0.7 }} flag={false} fairProb={0.52} />);
    const cell = screen.getByText('7/10');
    expect(cell.getAttribute('title')).toBe('Won 7 of his last 10 games at this line · market says 52%');
    expect(screen.queryByText(/cold/i)).toBeNull();
  });

  it('flags a big disagreement with the market', () => {
    render(<FormCell form={{ hits: 2, games: 10, rate: 0.2 }} flag={true} fairProb={0.5} />);
    expect(screen.getByText('2/10').className).toContain('text-[#fbbf24]');
    expect(screen.getByText('cold').getAttribute('title')).toContain('15+ points below');
  });

  it('shows a dash when there is no form', () => {
    render(<FormCell form={null} flag={false} fairProb={0.5} />);
    expect(screen.getByText('–')).toBeTruthy();
  });
});
