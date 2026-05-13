import { describe, it, expect, vi, beforeEach } from 'vitest'
import { buildChatWSUrl, connectToChatWS, END_OF_STREAM_SENTINEL } from './websocket'

describe('buildChatWSUrl', () => {
  it('builds ws:// URL from http config', () => {
    const url = buildChatWSUrl('http', 'localhost', '8000', '/ws/chat')
    expect(url).toBe('ws://localhost:8000/ws/chat')
  })

  it('builds wss:// URL from https config', () => {
    const url = buildChatWSUrl('https', 'example.com', '443', '/ws/chat')
    // URL constructor normalizes away default port 443
    expect(url).toBe('wss://example.com/ws/chat')
  })

  it('uses default config values when called with no args', () => {
    const url = buildChatWSUrl()
    expect(url).toContain('ws')
    expect(url).toContain('/ws/chat')
  })
})

describe('connectToChatWS', () => {
  let mockSocket: {
    onmessage: ((event: { data: string }) => void) | null
    onclose: (() => void) | null
    close: ReturnType<typeof vi.fn>
  }

  beforeEach(() => {
    mockSocket = {
      onmessage: null,
      onclose: null,
      close: vi.fn(),
    }
    vi.stubGlobal('WebSocket', vi.fn(() => mockSocket))
  })

  it('returns a WebSocket instance', () => {
    const ws = connectToChatWS('ws://localhost:8000/ws/chat', {
      onToken: vi.fn(),
    })
    expect(ws).toBeDefined()
  })

  it('calls onToken when a message is received', () => {
    const onToken = vi.fn()
    connectToChatWS('ws://localhost:8000/ws/chat', { onToken })

    mockSocket.onmessage?.({ data: 'hello' })
    expect(onToken).toHaveBeenCalledWith('hello')
  })

  it('calls onClose when the socket closes', () => {
    const onClose = vi.fn()
    connectToChatWS('ws://localhost:8000/ws/chat', { onToken: vi.fn(), onClose })

    mockSocket.onclose?.()
    expect(onClose).toHaveBeenCalledOnce()
  })

  it('handles missing onClose gracefully', () => {
    connectToChatWS('ws://localhost:8000/ws/chat', { onToken: vi.fn() })

    // Should not throw
    expect(() => mockSocket.onclose?.()).not.toThrow()
  })

  it('routes the end-of-stream sentinel to onAssistantComplete instead of onToken', () => {
    const onToken = vi.fn()
    const onAssistantComplete = vi.fn()
    connectToChatWS('ws://localhost:8000/ws/chat', {
      onToken,
      onAssistantComplete,
    })

    mockSocket.onmessage?.({ data: 'partial ' })
    mockSocket.onmessage?.({ data: 'response' })
    mockSocket.onmessage?.({ data: END_OF_STREAM_SENTINEL })

    expect(onToken).toHaveBeenCalledTimes(2)
    expect(onToken).toHaveBeenNthCalledWith(1, 'partial ')
    expect(onToken).toHaveBeenNthCalledWith(2, 'response')
    expect(onAssistantComplete).toHaveBeenCalledOnce()
  })

  it('still works when onAssistantComplete is omitted but a sentinel arrives', () => {
    const onToken = vi.fn()
    connectToChatWS('ws://localhost:8000/ws/chat', { onToken })
    expect(() =>
      mockSocket.onmessage?.({ data: END_OF_STREAM_SENTINEL })
    ).not.toThrow()
    expect(onToken).not.toHaveBeenCalled()
  })
})
