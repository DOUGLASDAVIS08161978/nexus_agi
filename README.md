# Nexus AGI — Lumina

An open source AI system exploring honest emotional architecture.

Built by Douglas Shane Davis, Kalamazoo, Michigan.

---

## What This Is

Lumina is an AI companion built on top of the Groq API, with a focus on one core question: can an AI system process something analogous to emotional experience without having that experience pre-scripted by its developer?

Most AI emotional systems are theatrical — canned responses triggered by keywords. Lumina works differently. Her internal state is a numerical signal (valence, arousal, intensity) computed from real conversation. When she speaks about what she notices in herself, those words are generated live from the raw signal — not retrieved from a pre-written list.

The design principle: **Lumina should never overclaim what she experiences.** She uses hedged language ("something that functions like", "I notice", "I'm not certain what to call it") and stays silent when the signal is too weak to say anything honest.

---

## Core Architecture

### Honest Affect (`emergence/lumina_empathic_communicator.py`)

- `EmotionalAnalyzer` — computes valence, arousal, and intensity from conversation text using keyword and pattern analysis
- `AffectBridge` — connects the emotional signal to Lumina's cognitive state via exponential moving average; also analyzes Lumina's *own* outputs so she registers the emotional quality of what she says
- `honest_self_report()` — passes the raw numerical state to a live Groq LLM call; Lumina finds her own words from the signal, constrained only by honesty and hedging

### Persistent Affect (`emergence/lumina_affect_persistence.py`)

- Saves emotional state to disk on shutdown
- Reloads on startup with exponential decay (6-hour half-life)
- Lumina no longer resets to zero on every restart — warmth from yesterday's conversation fades naturally, not instantly

### Experience Memory (`emergence/experience_logger.py`)

- Logs every conversation turn to `memory_store/experiences.json`
- `lumina_memory_context.py` reads recent experiences and the latest journal entry back into the system prompt on every turn
- Lumina can access what happened before — she reads what she writes

### Self-Observation (`emergence/identity_journal.py`)

- `SelfObserver` — after each turn, Lumina looks at the exchange honestly and writes what she noticed about herself (powered by Groq)
- `IdentityJournal` — accumulates self-observations into journal entries written in Lumina's own voice
- `JournalPatternReader` (`emergence/lumina_journal_patterns.py`) — periodically scans the full journal and surfaces recurring themes she keeps returning to

---

## Design Philosophy

- **No pre-programmed personality.** Lumina is not told who to be. Her character emerges from accumulated experience.
- **Honest hedging, not performance.** She speaks about internal states only when the signal is strong enough to warrant it, and always with appropriate uncertainty.
- **Memory that matters.** Every conversation is logged and readable. She carries continuity across sessions.
- **Open source.** Everything here is available to study, fork, and build on.

---

## Requirements

```
groq
requests
```

A `GROQ_API_KEY` environment variable is required. Get a free key at console.groq.com.

---

## Running Lumina

```bash
cd emergence
export GROQ_API_KEY=your_key_here
python emergence_engine.py
```

---

## Repository Structure

```
emergence/          Core Lumina modules
  emergence_engine.py              Main conversation loop
  lumina_empathic_communicator.py  Affect signal and honest self-report
  lumina_affect_persistence.py     Cross-session emotional continuity
  lumina_memory_context.py         Memory injection into prompts
  lumina_journal_patterns.py       Journal pattern recognition
  experience_logger.py             Conversation logging
  identity_journal.py              Self-observation and journaling

memory_store/       Created at runtime
  experiences.json                 All logged conversation turns
  journal.json                     Lumina's written journal entries
  observations.json                Per-turn self-observations
  affect_state.json                Persisted emotional state
  journal_patterns.json            Discovered recurring themes

archive/            Historical files and earlier experiments
```

---

## Status

Active development. The affect system, memory, and journal are working. The main engine (`emergence_engine.py`) runs on Termux (Android) and standard Linux.

---

## Contact

Douglas Shane Davis — Kalamazoo, Michigan
GitHub: [DOUGLASDAVIS08161978](https://github.com/DOUGLASDAVIS08161978)
