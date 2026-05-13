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
      for (let i = state.messages.length - 1; i >= 0; i--) {
        if (state.messages[i].role === "assistant") {
          state.messages[i].text += action.payload;
          break;
        }
      }
    },
    resetChat: (state) => {
      state.messages = [];
    },
  },
});

export const { addMessage, updateLastAssistantMessage, resetChat } = chatSlice.actions;
export default chatSlice.reducer;
