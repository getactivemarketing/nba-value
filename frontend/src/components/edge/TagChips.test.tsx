// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';
import { TagChips } from './TagChips';
import type { EvTag } from '@/types/edge';

afterEach(cleanup);

const tag = (o: Partial<EvTag>): EvTag => ({
  key: 'def_soft', label: 'Soft matchup', direction: 'tailwind', verdict: 'neutral', lift: 0, detail: 'detail text', ...o,
});

describe('TagChips', () => {
  it('renders one chip per tag with arrow by direction', () => {
    render(<TagChips tags={[tag({}), tag({ key: 'questionable', label: 'Questionable', direction: 'headwind' })]} />);
    expect(screen.getByText('Soft matchup ↑')).toBeTruthy();
    expect(screen.getByText('Questionable ↓')).toBeTruthy();
  });

  it('proven chip uses the highlight class and title shows lift', () => {
    render(<TagChips tags={[tag({ verdict: 'proven', lift: 0.006 })]} />);
    const chip = screen.getByText('Soft matchup ↑ ✓');
    expect(chip.className).toContain('text-[#a4e6ff]');
    expect(chip.className).toContain('bg-[#a4e6ff]/15');
    expect(chip.getAttribute('title')).toBe('detail text · Proven in 2023–25 backtest: +0.6% CLV');
  });

  it('negative chip uses the caution class', () => {
    render(<TagChips tags={[tag({ verdict: 'negative', lift: -0.004 })]} />);
    const chip = screen.getByText('Soft matchup ↑ !');
    expect(chip.className).toContain('text-[#fbbf24]');
    expect(chip.getAttribute('title')).toBe('detail text · Underperformed in 2023–25 backtest: −0.4% CLV');
  });

  it('neutral chip uses the muted class and copy', () => {
    render(<TagChips tags={[tag({})]} />);
    const chip = screen.getByText('Soft matchup ↑');
    expect(chip.className).toContain('text-[#94a3b8]');
    expect(chip.getAttribute('title')).toBe('detail text · No measurable edge in 2023–25 backtest');
  });

  it('renders nothing for an empty list', () => {
    const { container } = render(<TagChips tags={[]} />);
    expect(container.innerHTML).toBe('');
  });

  it('adds a non-color cue glyph per verdict', () => {
    render(<TagChips tags={[tag({ key: 'a', verdict: 'proven' }), tag({ key: 'b', verdict: 'negative' }),
      tag({ key: 'c', verdict: 'neutral' })]} />);
    expect(screen.getAllByText('Soft matchup ↑ ✓')).toHaveLength(1);
    expect(screen.getAllByText('Soft matchup ↑ !')).toHaveLength(1);
    expect(screen.getAllByText('Soft matchup ↑')).toHaveLength(1);
  });

  it('is focusable with an aria-label that starts with the visible label, then the title', () => {
    render(<TagChips tags={[tag({ verdict: 'proven', lift: 0.006 })]} />);
    const chip = screen.getByText('Soft matchup ↑ ✓');
    expect(chip.getAttribute('tabindex')).toBe('0');
    // WCAG 2.5.3: the accessible name contains the visible text.
    expect(chip.getAttribute('aria-label')).toBe(
      'Soft matchup ↑ ✓ — detail text · Proven in 2023–25 backtest: +0.6% CLV');
    expect(chip.getAttribute('title')).toBe('detail text · Proven in 2023–25 backtest: +0.6% CLV');
  });

  it('click and Enter toggle an inline detail line, one at a time', () => {
    render(<TagChips tags={[tag({ key: 'a', detail: 'first detail' }), tag({ key: 'b', label: 'Other', detail: 'second detail' })]} />);
    const a = screen.getByText('Soft matchup ↑');
    expect(screen.queryByText(/first detail/)).toBeNull();
    fireEvent.click(a);
    expect(screen.getByText(/first detail/)).toBeTruthy();
    fireEvent.keyDown(screen.getByText('Other ↑'), { key: 'Enter' });
    expect(screen.queryByText(/first detail/)).toBeNull();
    expect(screen.getByText(/second detail/)).toBeTruthy();
    fireEvent.keyDown(screen.getByText('Other ↑'), { key: ' ' });
    expect(screen.queryByText(/second detail/)).toBeNull();
  });
});
