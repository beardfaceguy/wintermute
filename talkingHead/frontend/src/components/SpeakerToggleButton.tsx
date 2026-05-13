// src/components/SpeakerToggleButton.tsx
import React, { memo } from "react";
import { Volume2, VolumeX } from "lucide-react";

type SpeakerToggleButtonProps = {
  enabled: boolean;
  available: boolean;
  onToggle: () => void;
};

const SpeakerToggleButton: React.FC<SpeakerToggleButtonProps> = ({
  enabled,
  available,
  onToggle,
}) => {
  if (!available) return null;
  return (
    <button
      className={`chat-button ${enabled ? "" : "muted"}`}
      onClick={onToggle}
      title={enabled ? "Mute voice output" : "Enable voice output"}
      aria-label={enabled ? "Mute voice output" : "Enable voice output"}
      aria-pressed={enabled}
    >
      {enabled ? <Volume2 size={20} /> : <VolumeX size={20} />}
    </button>
  );
};

export default memo(SpeakerToggleButton);
