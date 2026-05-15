"use client";

export interface FrontendErrorEvent {
  source: string;
  message: string;
  digest?: string;
  path?: string;
  timestamp: string;
}

export function reportFrontendError(event: Omit<FrontendErrorEvent, "timestamp">) {
  const payload: FrontendErrorEvent = {
    ...event,
    timestamp: new Date().toISOString(),
  };

  console.error("[qgg.frontend_error]", payload);
}
