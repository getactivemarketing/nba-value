// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { edgeApi } from '@/lib/edgeApi';
import { EmailSignup } from './EmailSignup';

afterEach(() => { cleanup(); vi.restoreAllMocks(); });

const submit = (email: string) => {
  fireEvent.change(screen.getByLabelText(/email address/i), { target: { value: email } });
  fireEvent.click(screen.getByRole('button', { name: /notify me/i }));
};

describe('EmailSignup', () => {
  it('sends the email with its page source and thanks the visitor', async () => {
    const spy = vi.spyOn(edgeApi, 'signup').mockResolvedValue(undefined);
    render(<EmailSignup source="edge_home" />);
    submit('fan@example.com');
    await waitFor(() => expect(screen.getByText(/you're on the list/i)).toBeTruthy());
    expect(spy).toHaveBeenCalledWith('fan@example.com', 'edge_home', '');
  });

  it('shows an error and keeps the form when the request fails', async () => {
    vi.spyOn(edgeApi, 'signup').mockRejectedValue(new Error('boom'));
    render(<EmailSignup source="edge_home" />);
    submit('fan@example.com');
    await waitFor(() => expect(screen.getByRole('alert')).toBeTruthy());
    expect(screen.getByLabelText(/email address/i)).toBeTruthy();
  });

  it('hides the honeypot field from people', () => {
    render(<EmailSignup source="edge_home" />);
    const trap = document.querySelector('input[name="website"]') as HTMLInputElement;
    expect(trap.tabIndex).toBe(-1);
    expect(trap.closest('[aria-hidden="true"]')).toBeTruthy();
  });
});
