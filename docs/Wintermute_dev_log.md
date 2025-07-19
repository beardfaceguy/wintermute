
## 📌 DESIGN DECISIONS

### DEC-01: Use FastAPI for backend to support token streaming
**Date:** 2025-04-15  
**Rationale:** Django not optimal for async/token stream; FastAPI simplifies agent interaction.  
**Linked CP:** N/A

### DEC-02: Memory split into Live and Cold tiers  
**Date:** 2025-04-19  
**Rationale:** Mitigate hallucination and strategic entropy  
**Linked CP:** CP-0001


## 🔄 CHANGE PROPOSALS

### CP-0001: Introduce memory lifecycle management (Live ➜ Cold)  
**Status:** Approved  
**Date Proposed:** 2025-04-19  
**Summary:** Add promotion workflow gated by sanity checks  
**Linked DEC:** DEC-02  
**Auditor Input:** Freud, planned expansion to Jung/Adler

### CP-0002: Add `thVoice` browser interface toggle to TalkingHead  
**Status:** In Progress  
**Date Proposed:** 2025-07-10  
**Summary:** Integrate voice interface into web UI as optional frontend feature  
**Linked DEC:** TBD  
**Tasks:** 
    [x] Mic toggle 
    [x] STT routing to backend
    [x] Whisper.cpp transcription
    [ ] Inject transcription as simulated user message
    [ ] Use Piper to read aloud LLM responses