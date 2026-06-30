"""Host-agnostic serving: provision a trained model on a chosen backend.

Provisioning only — inference clients are reused from eval.model (vLLM/Ollama
speak OpenAI-compat via OpenAICompatBackend), never duplicated here.
"""
