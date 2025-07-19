// src/components/VoiceToggleButton.tsx
import React, { useState, memo  } from "react";
import useVoiceRecorder from "../hooks/useVoiceRecorder";
import { Mic, StopCircle } from "lucide-react";

console.log("🔧 VoiceToggleButton component initialized");

type VoiceToggleButtonProps = {
  onVoiceSubmit: (text: string) => void;
};

const VoiceToggleButton: React.FC<VoiceToggleButtonProps> = ({ onVoiceSubmit }) => {
  const [recording, setRecording] = useState(false);

  const { onStartRecording, onStopRecording } = useVoiceRecorder({
    onTranscriptionResult: (text: string) => {
      console.log("🎤 Transcribed:", text);
      onVoiceSubmit(text);
    },
  });

  const handleClick = () => {
    if (recording) {
      onStopRecording();
    } else {
      onStartRecording();
    }
    setRecording(!recording);
  };

  return (
    <button
      className={`chat-button ${recording ? "recording" : ""}`}
      onClick={handleClick}
      title={recording ? "Stop Recording" : "Start Recording"}
    >
      {recording ? <StopCircle size={20} /> : <Mic size={20} />}
    </button>
  );
};

export default memo(VoiceToggleButton);
