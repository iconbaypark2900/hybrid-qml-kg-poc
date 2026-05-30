'use client';

import { QueryClient, QueryClientProvider, type DefaultOptions } from '@tanstack/react-query';
import { type ReactNode, useState } from 'react';
import { ApiError } from '@/lib/api';

const defaultOptions: DefaultOptions = {
  queries: {
    // Manifest + status are stable for a few seconds at a time. Stale time
    // longer than that (e.g. 30s) means the dashboard doesn't hammer /status
    // when multiple components subscribe. The poll on individual hooks still
    // runs at their own cadence.
    staleTime: 30_000,
    gcTime: 5 * 60_000,
    retry: (failureCount, error) => {
      // Don't retry auth or 4xx (except 429); do retry network + 5xx + 429.
      if (error instanceof ApiError) {
        if (error.code === 'auth_missing' || error.code === 'auth_invalid') return false;
        if (error.status >= 400 && error.status < 500 && error.status !== 429) return false;
      }
      return failureCount < 2;
    },
    refetchOnWindowFocus: false,
    refetchOnReconnect: true,
  },
  mutations: {
    retry: false,
  },
};

export function QueryProvider({ children }: { children: ReactNode }) {
  // useState ensures the QueryClient is created once per browser session,
  // not per render. SSR-safe: never created on the server.
  const [client] = useState(() => new QueryClient({ defaultOptions }));
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}
