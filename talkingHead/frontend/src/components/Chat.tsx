import React, { useEffect, useRef, useState } from 'react'
import { useDispatch, useSelector } from 'react-redux'
import { type RootState } from '../store'
import { addMessage } from '../store/chatSlice'

const Chat: React.FC = () => {
  const dispatch = useDispatch()
  const messages = useSelector((state: RootState) => state.chat.messages)
  const [input, setInput] = useState('')
  const ws = useRef<WebSocket | null>(null)

  useEffect(() => {
    ws.current = new WebSocket('ws://localhost:8000/ws/chat')
    ws.current.onmessage = (event) => {
      dispatch(addMessage({ sender: 'bot', content: event.data }))
    }
    return () => ws.current?.close()
  }, [dispatch])

  const sendMessage = () => {
    if (ws.current && input.trim() !== '') {
      ws.current.send(input)
      dispatch(addMessage({ sender: 'user', content: input }))
      setInput('')
    }
  }

  return (
    <div className="flex flex-col h-screen p-4">
      <div className="flex-1 overflow-y-auto space-y-2">
        {messages.map((msg, i) => (
          <div key={i} className={msg.sender === 'user' ? 'text-right' : 'text-left'}>
            <span className="inline-block p-2 rounded bg-gray-200 dark:bg-gray-700">
              {msg.content}
            </span>
          </div>
        ))}
      </div>
      <div className="mt-4 flex">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
          className="flex-1 p-2 border rounded"
          placeholder="Say something..."
        />
        <button onClick={sendMessage} className="ml-2 px-4 py-2 bg-blue-500 text-white rounded">
          Send
        </button>
      </div>
    </div>
  )
}

export default Chat
