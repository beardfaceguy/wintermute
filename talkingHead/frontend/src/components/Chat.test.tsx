import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { Provider } from 'react-redux'
import { configureStore } from '@reduxjs/toolkit'
import chatReducer from '../store/chatSlice'
import Chat from './Chat'

vi.mock('../hooks/useChatSocket', () => ({
  default: () => ({ sendMessage: vi.fn() }),
}))

vi.mock('./VoiceToggleButton', () => ({
  default: () => <button data-testid="voice-btn">Voice</button>,
}))

function renderWithStore(preloadedMessages: Array<{ role: 'user' | 'assistant'; text: string }> = []) {
  const store = configureStore({
    reducer: { chat: chatReducer },
    preloadedState: { chat: { messages: preloadedMessages } },
  })
  return { store, ...render(<Provider store={store}><Chat /></Provider>) }
}

describe('Chat component', () => {
  it('renders the text input and send button', () => {
    renderWithStore()
    expect(screen.getByPlaceholderText('Type a message...')).toBeInTheDocument()
    expect(screen.getByText('Send')).toBeInTheDocument()
  })

  it('renders existing messages from the store', () => {
    renderWithStore([
      { role: 'user', text: 'hello' },
      { role: 'assistant', text: 'hi there' },
    ])
    expect(screen.getByText('hello')).toBeInTheDocument()
    expect(screen.getByText('hi there')).toBeInTheDocument()
  })

  it('distinguishes user and assistant message styling', () => {
    renderWithStore([
      { role: 'user', text: 'user-msg' },
      { role: 'assistant', text: 'assistant-msg' },
    ])
    const userMsg = screen.getByText('user-msg')
    const assistantMsg = screen.getByText('assistant-msg')
    expect(userMsg.className).toContain('chat-user')
    expect(assistantMsg.className).toContain('chat-assistant')
  })

  it('clears the input after sending', () => {
    renderWithStore()
    const input = screen.getByPlaceholderText('Type a message...') as HTMLTextAreaElement

    fireEvent.change(input, { target: { value: 'test message' } })
    expect(input.value).toBe('test message')

    fireEvent.click(screen.getByText('Send'))
    expect(input.value).toBe('')
  })

  it('does not send an empty message', () => {
    const { store } = renderWithStore()

    fireEvent.click(screen.getByText('Send'))

    // Store should still have no messages
    expect(store.getState().chat.messages).toHaveLength(0)
  })

  it('sends on Enter key (without Shift)', () => {
    renderWithStore()
    const input = screen.getByPlaceholderText('Type a message...')

    fireEvent.change(input, { target: { value: 'enter test' } })
    fireEvent.keyDown(input, { key: 'Enter', shiftKey: false })

    expect((input as HTMLTextAreaElement).value).toBe('')
  })

  it('does not send on Shift+Enter (allows newline)', () => {
    renderWithStore()
    const input = screen.getByPlaceholderText('Type a message...')

    fireEvent.change(input, { target: { value: 'line1' } })
    fireEvent.keyDown(input, { key: 'Enter', shiftKey: true })

    expect((input as HTMLTextAreaElement).value).toBe('line1')
  })

  it('renders the voice toggle button', () => {
    renderWithStore()
    expect(screen.getByTestId('voice-btn')).toBeInTheDocument()
  })
})
