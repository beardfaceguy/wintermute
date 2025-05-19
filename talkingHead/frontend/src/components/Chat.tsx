import React, { useEffect, useRef, useState } from 'react'
import { useDispatch, useSelector } from 'react-redux'
import { type RootState } from '../store'
import { addMessage } from '../store/chatSlice'

const Chat: React.FC = () => {
  const dispatch = useDispatch()
  const messages = useSelector((state: RootState) => state.chat.messages)
  const [input, setInput] = useState('')
  const ws = useRef<WebSocket | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    ws.current = new WebSocket('ws://localhost:8000/ws/chat')
    ws.current.onmessage = (event) => {
      dispatch(addMessage({ sender: 'bot', content: event.data }))
    }
    return () => ws.current?.close()
  }, [dispatch])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const sendMessage = () => {
    if (ws.current && input.trim() !== '') {
      ws.current.send(input)
      dispatch(addMessage({ sender: 'user', content: input }))
      setInput('')
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') sendMessage()
  }

  return (
    <div className="flex flex-col h-screen bg-white">
      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {messages.map((msg, idx) => (
          <div
            key={idx}
            className={`max-w-[75%] p-2 rounded-lg shadow-sm whitespace-pre-wrap break-words ${
              msg.sender === 'user'
                ? 'ml-auto bg-blue-100 text-right'
                : 'mr-auto bg-gray-200 text-left'
            }`}
          >
            {msg.content}
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      <div className="flex gap-2 p-4 border-t bg-gray-50">
        <input
          type="text"
          className="flex-1 border rounded px-3 py-2 text-sm"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyPress}
          placeholder="Say something..."
        />
        <button
          onClick={sendMessage}
          className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
        >
          Send
        </button>
      </div>
    </div>
  )
}

export default Chat
