/*
 * /src/store/chatSlice.ts
 */

import { createSlice, type PayloadAction } from '@reduxjs/toolkit';
import {debugLog} from '../utils/debug';
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
      debugLog("chatSlice.addMessage: Adding message to chat:", action.payload);
      state.messages.push(action.payload);
    },
    updateLastAssistantMessage: (state, action: PayloadAction<string>) => {
      debugLog("chatSlice.updateLastAssistantMessage: Updating last assistant message with:", action.payload);
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
