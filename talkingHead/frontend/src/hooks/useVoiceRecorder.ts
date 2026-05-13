// src/hooks/useVoiceRecorder.ts
import { useCallback, useRef, useState } from "react";
import { debugLog } from "../utils/debug";
import rawconfig from '../../../../config/shared_api_config.json';

const voiceUrl = (() => {
  const cfg = rawconfig as { web_interface: { scheme: string; host: string; port: number } };
  const host = cfg.web_interface.host === "localhost" && typeof window !== "undefined" && window.location.hostname !== "localhost"
    ? window.location.hostname
    : cfg.web_interface.host;
  return `${cfg.web_interface.scheme}://${host}:${cfg.web_interface.port}/api/chat/voice`;
})();

type UseVoiceRecorderProps = {
  onTranscriptionResult: (text: string) => void;
};

const useVoiceRecorder = ({ onTranscriptionResult }: UseVoiceRecorderProps) => {
  debugLog("useVoiceRecorder initialized");

  const [isRecording, setIsRecording] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  const sendAudio = useCallback(async (audioBlob: Blob) => {
    const formData = new FormData();
    formData.append("file", audioBlob, "audio.wav");

    try {
      const response = await fetch(voiceUrl, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Failed to upload audio");
      }

      const result = await response.json();
      console.log("📥 Full backend response:", result);

      if (result.transcript) {
        console.log("📤 Passing transcript to callback:", result.transcript);
        onTranscriptionResult(result.transcript);
      } else {
        console.warn("⚠️ No transcript found in response");
      }
    } catch (error) {
      console.error("❌ Error sending audio:", error);
    }
  }, [onTranscriptionResult]);

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: "audio/wav" });
        sendAudio(audioBlob);
        stream.getTracks().forEach((track) => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
      console.log("🔴 Recording started");
    } catch (error) {
      console.error("❌ Error starting recording:", error);
    }
  }, [sendAudio]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      console.log("⏹️ Recording stopped");
    }
  }, [isRecording]);

  return {
    isRecording,
    onStartRecording: startRecording,
    onStopRecording: stopRecording,
  };
};

export default useVoiceRecorder;
