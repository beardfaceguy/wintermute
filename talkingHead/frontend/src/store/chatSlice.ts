/*
 * /src/store/chatSlice.ts
 */

import { createSlice, type PayloadAction } from '@reduxjs/toolkit';

export interface ChatMessage {
  role: 'user' | 'assistant';
  text: string;
}

interface ChatState {
  messages: ChatMessage[];
}

const initialState: ChatState = {
  messages: [],
};

const chatSlice = createSlice({
  name: 'chat',
  initialState,
  reducers: {
    addMessage: (state, action: PayloadAction<ChatMessage>) => {
      state.messages.push(action.payload);
    },
    updateLastAssistantMessage: (state, action: PayloadAction<string>) => {
      const lastMsg = [...state.messages].reverse().find((msg) => msg.role === "assistant");
      if (lastMsg) {
        lastMsg.text += action.payload;
      }
    },
    resetChat: (state) => {
      state.messages = [];
    },
  },
});

export const { addMessage, updateLastAssistantMessage, resetChat } = chatSlice.actions;
export default chatSlice.reducer;
