// utils/debug.ts
export function debugLog(...args: unknown[]) {

  if (import.meta.env.MODE === 'development') {
    const stack = new Error().stack;
    const callerLine = stack?.split('\n')[2] ?? '';
    const match = callerLine.match(/\((.*):(\d+):(\d+)\)/);

    const fileInfo = match
        ? `${match[1].split('/').slice(-2).join('/')}:${match[2]}`
        : 'unknown';

    console.log(`[DEBUG][${fileInfo}]`, ...args);
  }
}
