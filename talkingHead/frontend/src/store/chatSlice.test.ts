import { describe, it, expect } from 'vitest'
import chatReducer, {
  addMessage,
  updateLastAssistantMessage,
  resetChat,
  type ChatMessage,
} from './chatSlice'

interface ChatState {
  messages: ChatMessage[]
}

const empty: ChatState = { messages: [] }

describe('chatSlice', () => {
  it('starts with an empty message list', () => {
    const state = chatReducer(undefined, { type: '@@INIT' })
    expect(state.messages).toEqual([])
  })

  it('addMessage appends a user message', () => {
    const state = chatReducer(empty, addMessage({ role: 'user', text: 'hello' }))
    expect(state.messages).toHaveLength(1)
    expect(state.messages[0]).toEqual({ role: 'user', text: 'hello' })
  })

  it('addMessage appends an assistant message', () => {
    const prev: ChatState = {
      messages: [{ role: 'user', text: 'hi' }],
    }
    const state = chatReducer(prev, addMessage({ role: 'assistant', text: '' }))
    expect(state.messages).toHaveLength(2)
    expect(state.messages[1].role).toBe('assistant')
  })

  it('updateLastAssistantMessage appends to the last assistant message', () => {
    const prev: ChatState = {
      messages: [
        { role: 'user', text: 'hi' },
        { role: 'assistant', text: 'hel' },
      ],
    }
    const state = chatReducer(prev, updateLastAssistantMessage('lo'))
    expect(state.messages[1].text).toBe('hello')
  })

  it('updateLastAssistantMessage finds the correct assistant among multiple', () => {
    const prev: ChatState = {
      messages: [
        { role: 'user', text: 'q1' },
        { role: 'assistant', text: 'a1' },
        { role: 'user', text: 'q2' },
        { role: 'assistant', text: 'a2-partial' },
      ],
    }
    const state = chatReducer(prev, updateLastAssistantMessage('-done'))
    expect(state.messages[1].text).toBe('a1')
    expect(state.messages[3].text).toBe('a2-partial-done')
  })

  it('updateLastAssistantMessage is a no-op with no assistant messages', () => {
    const prev: ChatState = {
      messages: [{ role: 'user', text: 'hi' }],
    }
    const state = chatReducer(prev, updateLastAssistantMessage('token'))
    expect(state.messages).toHaveLength(1)
    expect(state.messages[0].text).toBe('hi')
  })

  it('resetChat clears all messages', () => {
    const prev: ChatState = {
      messages: [
        { role: 'user', text: 'hi' },
        { role: 'assistant', text: 'hey' },
      ],
    }
    const state = chatReducer(prev, resetChat())
    expect(state.messages).toEqual([])
  })

  it('handles rapid token streaming without corruption', () => {
    let state: ChatState = {
      messages: [
        { role: 'user', text: 'go' },
        { role: 'assistant', text: '' },
      ],
    }
    const tokens = ['The', ' quick', ' brown', ' fox']
    for (const t of tokens) {
      state = chatReducer(state, updateLastAssistantMessage(t))
    }
    expect(state.messages[1].text).toBe('The quick brown fox')
  })
})
