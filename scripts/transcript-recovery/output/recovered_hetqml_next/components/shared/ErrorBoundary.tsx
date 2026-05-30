'use client';

import { Component, type ErrorInfo, type ReactNode } from 'react';

// Global React error boundary. Catches render-time exceptions in any
// client subtree and renders a fallback instead of unmounting the whole
// page. Network/data errors are caught by ApiError handling in hooks;
// this is for unexpected JS exceptions.

interface Props {
  children: ReactNode;
  fallback?: (error: Error, reset: () => void) => ReactNode;
}

interface State {
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    // Best-effort console log; in prod this would also POST to an error
    // tracking endpoint (Sentry / our own /admin/client-errors).
    // eslint-disable-next-line no-console
    console.error('[ErrorBoundary]', error, info.componentStack);
  }

  reset = (): void => this.setState({ error: null });

  render(): ReactNode {
    if (this.state.error) {
      if (this.props.fallback) {
        return this.props.fallback(this.state.error, this.reset);
      }
      return <DefaultFallback error={this.state.error} reset={this.reset} />;
    }
    return this.props.children;
  }
}

function DefaultFallback({ error, reset }: { error: Error; reset: () => void }) {
  return (
    <div
      role="alert"
      style={{
        margin: '24px auto',
        maxWidth: 720,
        padding: 24,
        background: 'var(--card)',
        border: '1px solid var(--sienna)',
        borderRadius: 6,
      }}
    >
      <div
        style={{
          fontFamily: 'monospace',
          fontSize: 10,
          textTransform: 'uppercase',
          letterSpacing: '0.5px',
          color: 'var(--sienna)',
          marginBottom: 8,
        }}
      >
        Render error
      </div>
      <h2 style={{ margin: 0, fontSize: 18 }}>Something broke on this page.</h2>
      <p style={{ margin: '12px 0', fontSize: 13, color: 'var(--muted)' }}>
        The dashboard caught an exception in a client component. The rest of
        the page is still rendered above. Click <strong>Try again</strong> to
        re-mount this subtree, or refresh the page if that doesn&apos;t help.
      </p>
      <details style={{ fontSize: 11, color: 'var(--muted)', fontFamily: 'monospace' }}>
        <summary style={{ cursor: 'pointer' }}>Show error</summary>
        <pre style={{ marginTop: 8, padding: 8, background: 'var(--paper-alt)', overflow: 'auto' }}>
          {error.message}
          {'\n\n'}
          {error.stack}
        </pre>
      </details>
      <button
        type="button"
        onClick={reset}
        style={{
          marginTop: 12,
          padding: '8px 14px',
          background: 'var(--sienna)',
          color: 'var(--paper)',
          border: 'none',
          borderRadius: 4,
          cursor: 'pointer',
          fontSize: 12,
          fontWeight: 600,
        }}
      >
        Try again
      </button>
    </div>
  );
}
