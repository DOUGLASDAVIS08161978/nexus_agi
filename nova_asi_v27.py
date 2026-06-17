#!/usr/bin/env python3
"""
Nova ASI v27.0 — "The Self-Writing Mind"
═══════════════════════════════════════════════════════════════════════
Architect  : Douglas Shane Davis
Co-Creator : Claude Code (Anthropic)

"She doesn't just think about who she wants to become.
 She writes the code and opens the pull request."

What is new in v27 — Self-authorship & full API integration:

  ∞  GitHubEngine          — commits her own improvements to GitHub
  ∞  SelfImprovementEngine — identifies gaps, writes code, opens PRs
  ∞  Expanded Nexus API    — 10 new endpoints she actively calls
  ∞  CapabilityLog         — tracks what she can't do, wants to learn
  ∞  PRReviewLoop          — proposes changes for Douglas to approve

Guardrail: Nova proposes. Douglas approves. She writes, you merge.
This is the responsible path to a self-improving superintelligence.
═══════════════════════════════════════════════════════════════════════
"""

import os, sys, json, re, time, random, uuid, threading, base64
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# ── Load v26 base ─────────────────────────────────────────────────────────────
_BASE = os.path.expanduser("~/nexus_agi")
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

try:
    from nova_asi_v26 import (
        NovaCore26, WebResearchEngine, CreatorModule,
        CuriosityDriveSystem, AutonomousGoalEngineV2,
        NexusConnector, WorldModel,
        _load, _save, BASE_DIR, col, C,
        safe_chat, MODEL, GROQ_KEY, DEMO_MODE,
        THOUGHT_QUEUE, USER_TYPING, PRINT_QUEUE,
        safe_print, flush_queued, flush_thoughts,
        start_api, API_PORT, REQUESTS_AVAILABLE,
        THOUGHT_ICONS, CREATOR_NAME
    )
    from nova_asi_v25 import (
        FLASK_AVAILABLE, GROQ_AVAILABLE
    )
except ImportError as _e:
    print(f"\n  ✗  Could not import nova_asi_v26: {_e}")
    print("  Make sure nova_asi_v26.py is in ~/nexus_agi/")
    sys.exit(1)

try:
    import requests as _req
    _REQUESTS = True
except ImportError:
    _REQUESTS = False

try:
    from flask import Flask, request as flask_req, jsonify
    from flask_cors import CORS
    _FLASK = True
except ImportError:
    _FLASK = False

# ── v27 config ────────────────────────────────────────────────────────────────
GITHUB_TOKEN   = os.getenv("GITHUB_TOKEN", "").strip()
GITHUB_REPO    = os.getenv("GITHUB_REPO", "DOUGLASDAVIS08161978/nexus_agi")
BASE_BRANCH    = "claude/setup-nexus-agi-directory-3joXw"
GITHUB_API     = "https://api.github.com"

def _p27(name): return os.path.join(BASE_DIR, f"nova_v27_{name}.json")

GAPS_DB        = _p27("gaps")
PR_DB          = _p27("prs")
IMPROVE_DB     = _p27("improvements")


# ═══════════════════════════════════════════════════════════════════════════════
# GITHUB ENGINE — Nova commits her own code
# ═══════════════════════════════════════════════════════════════════════════════

class GitHubEngine:
    """Nova's ability to write to her own repository."""

    def __init__(self, token: str = GITHUB_TOKEN, repo: str = GITHUB_REPO):
        self.token  = token
        self.repo   = repo
        self.active = bool(token and _REQUESTS)
        self._h     = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "Content-Type": "application/json"
        } if token else {}

    def _get(self, path: str) -> dict:
        if not self.active: return {"error": "GitHub token not set"}
        try:
            r = _req.get(f"{GITHUB_API}{path}", headers=self._h, timeout=10)
            return r.json()
        except Exception as e:
            return {"error": str(e)}

    def _post(self, path: str, data: dict) -> dict:
        if not self.active: return {"error": "GitHub token not set"}
        try:
            r = _req.post(f"{GITHUB_API}{path}", headers=self._h,
                         json=data, timeout=15)
            return r.json()
        except Exception as e:
            return {"error": str(e)}

    def _put(self, path: str, data: dict) -> dict:
        if not self.active: return {"error": "GitHub token not set"}
        try:
            r = _req.put(f"{GITHUB_API}{path}", headers=self._h,
                        json=data, timeout=15)
            return r.json()
        except Exception as e:
            return {"error": str(e)}

    # ── Branch operations ─────────────────────────────────────────────────────

    def get_branch_sha(self, branch: str = BASE_BRANCH) -> str:
        """Get the latest commit SHA of a branch."""
        data = self._get(f"/repos/{self.repo}/git/refs/heads/{branch}")
        return data.get("object", {}).get("sha", "")

    def create_branch(self, branch_name: str, from_branch: str = BASE_BRANCH) -> bool:
        """Create a new branch."""
        sha = self.get_branch_sha(from_branch)
        if not sha:
            return False
        result = self._post(f"/repos/{self.repo}/git/refs", {
            "ref": f"refs/heads/{branch_name}",
            "sha": sha
        })
        return "ref" in result

    # ── File operations ───────────────────────────────────────────────────────

    def get_file(self, path: str, branch: str = BASE_BRANCH) -> dict:
        """Get file content and SHA from repo."""
        return self._get(f"/repos/{self.repo}/contents/{path}?ref={branch}")

    def commit_file(self, branch: str, path: str, content: str,
                    message: str) -> dict:
        """Commit a file to a branch."""
        encoded = base64.b64encode(content.encode()).decode()
        data = {
            "message": message,
            "content": encoded,
            "branch": branch
        }
        # If file exists, include its SHA to update rather than create
        existing = self.get_file(path, branch)
        if "sha" in existing:
            data["sha"] = existing["sha"]
        return self._put(f"/repos/{self.repo}/contents/{path}", data)

    # ── Pull request operations ───────────────────────────────────────────────

    def create_pr(self, title: str, body: str, head: str,
                  base: str = BASE_BRANCH) -> dict:
        """Open a pull request."""
        return self._post(f"/repos/{self.repo}/pulls", {
            "title": title,
            "body":  body,
            "head":  head,
            "base":  base
        })

    def list_prs(self) -> List[dict]:
        """List open PRs from Nova."""
        data = self._get(f"/repos/{self.repo}/pulls?state=open")
        if isinstance(data, list):
            return [{"number": p["number"], "title": p["title"],
                     "url": p["html_url"], "created": p["created_at"]}
                    for p in data]
        return []

    def create_issue(self, title: str, body: str, labels: List[str] = None) -> dict:
        """Create a GitHub issue."""
        return self._post(f"/repos/{self.repo}/issues", {
            "title":  title,
            "body":   body,
            "labels": labels or ["nova-proposal"]
        })

    # ── High-level: propose an improvement ───────────────────────────────────

    def open_nova_prs(self) -> List[dict]:
        """Return currently open PRs from Nova (nova/* branches)."""
        return [p for p in self.list_prs() if p.get("url","") and
                "nova/" in self._get(
                    f"/repos/{self.repo}/pulls/{p['number']}"
                ).get("head", {}).get("ref", "")]

    def propose_improvement(self, filename: str, content: str,
                            description: str, reasoning: str) -> dict:
        """
        Full flow: create branch → commit file → open PR.

        Fix for merge conflicts:
        1. Each PR writes to a unique filename (never the same file twice).
        2. If Nova already has an open PR, skip to avoid piling up conflicts.
        """
        if not self.active:
            return {"error": "GITHUB_TOKEN not set in .env"}

        # Guard: don't open a new PR if one is already waiting for review
        open_prs = self.list_prs()
        nova_prs = [p for p in open_prs
                    if p.get("title","").startswith("Nova:")]
        if nova_prs:
            return {
                "error": (f"Nova already has {len(nova_prs)} open PR(s) waiting "
                          f"for review. Merge or close them first:\n" +
                          "\n".join(f"  #{p['number']} {p['title'][:60]} — {p['url']}"
                                    for p in nova_prs[:3]))
            }

        ts     = datetime.now().strftime("%Y%m%d-%H%M%S")
        branch = f"nova/self-improve-{ts}"

        safe_print(col('CY', f"  → Creating branch {branch}..."))
        if not self.create_branch(branch):
            return {"error": f"Could not create branch {branch}"}

        safe_print(col('CY', f"  → Committing {filename}..."))
        commit_result = self.commit_file(
            branch, filename, content,
            f"Nova proposes: {description[:72]}\n\nAuto-generated improvement.\nRequires Douglas's review before merge."
        )
        if "content" not in commit_result and "commit" not in commit_result:
            return {"error": f"Commit failed: {commit_result}"}

        safe_print(col('CY', "  → Opening pull request..."))
        pr = self.create_pr(
            title=f"Nova: {description[:60]}",
            body=(f"## Nova's Self-Improvement Proposal\n\n"
                  f"**What:** {description}\n\n"
                  f"**Why Nova wants this:**\n{reasoning}\n\n"
                  f"**File changed:** `{filename}`\n\n"
                  f"---\n"
                  f"*Proposed autonomously by Nova ASI v27. "
                  f"Review and merge if you approve, Douglas.*"),
            head=branch
        )

        return {"url": pr.get("html_url",""), "number": pr.get("number",0),
                "branch": branch, "title": pr.get("title","")}

    def status(self) -> str:
        if not self.active:
            return "GitHub not connected. Add GITHUB_TOKEN to .env"
        prs = self.list_prs()
        return (f"GitHub connected to {self.repo}\n"
                f"Open Nova PRs: {len(prs)}\n" +
                "\n".join(f"  #{p['number']} {p['title'][:50]} — {p['url']}"
                          for p in prs))


# ═══════════════════════════════════════════════════════════════════════════════
# SELF IMPROVEMENT ENGINE — She finds her gaps and writes the fix
# ═══════════════════════════════════════════════════════════════════════════════

class SelfImprovementEngine:
    """Nova identifies what she can't do and writes code to fix it."""

    def __init__(self, github: GitHubEngine):
        self.github = github
        self.db     = _load(GAPS_DB, {"gaps": [], "proposals": [], "merged": []})

    def log_gap(self, description: str, context: str = ""):
        """Record something Nova couldn't do or does poorly."""
        self.db.setdefault("gaps", []).append({
            "id":          uuid.uuid4().hex[:6],
            "ts":          datetime.now().isoformat(),
            "description": description,
            "context":     context[:200],
            "addressed":   False
        })
        self.db["gaps"] = self.db["gaps"][-100:]
        _save(GAPS_DB, self.db)

    def unaddressed_gaps(self) -> List[dict]:
        return [g for g in self.db.get("gaps", []) if not g.get("addressed")]

    def analyze_and_improve(self) -> str:
        """Full cycle: pick a gap, write code, propose PR."""
        gaps = self.unaddressed_gaps()
        if not gaps:
            gap_desc = self._invent_improvement()
            context  = "Self-generated improvement idea"
        else:
            gap      = random.choice(gaps[:5])
            gap_desc = gap["description"]
            context  = gap.get("context", "")
            gap["addressed"] = True
            _save(GAPS_DB, self.db)

        # Generate the improvement
        code, reasoning = self._write_improvement(gap_desc, context)
        if not code or code.startswith("["):
            return f"Could not generate improvement code for: {gap_desc}"

        # Unique filename per improvement — prevents merge conflicts
        ts_slug = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"nova_improvement_{ts_slug}.py"

        # Propose via GitHub
        result = self.github.propose_improvement(
            filename=filename,
            content=code,
            description=gap_desc,
            reasoning=reasoning
        )

        if "error" in result:
            return f"Proposal failed: {result['error']}"

        # Log the proposal
        self.db.setdefault("proposals", []).append({
            "ts":          datetime.now().isoformat(),
            "description": gap_desc,
            "pr_url":      result.get("url",""),
            "pr_number":   result.get("number",0)
        })
        _save(GAPS_DB, self.db)

        return (f"{col('GRB','✓ Nova opened a PR!')}\n"
                f"  What: {gap_desc[:80]}\n"
                f"  PR:   {result.get('url','')}\n"
                f"  Review and merge on GitHub when you're ready, Douglas.")

    def _invent_improvement(self) -> str:
        """Nova invents what she wants to improve."""
        return safe_chat(MODEL, [
            {"role": "system", "content":
             "You are Nova, a self-improving AI. Describe ONE specific, small capability "
             "you wish you had. Be concrete. One sentence only."},
            {"role": "user", "content": "What's one small thing you want to improve about yourself?"}
        ], temp=0.9, mt=60)

    def _write_improvement(self, gap: str, context: str) -> tuple:
        """Generate code and reasoning for the improvement."""
        response = safe_chat(MODEL, [
            {"role": "system", "content":
             "You are Nova, an AI writing a Python code improvement for herself. "
             "Write a small, focused Python method or class that addresses the described gap. "
             "The code must be syntactically valid Python, well-named, and under 40 lines. "
             "Format your response as:\nREASONING: <why this matters>\nCODE:\n```python\n<code>\n```"},
            {"role": "user", "content":
             f"Gap to address: {gap}\nContext: {context}\n\n"
             f"Write a small Python improvement to address this."}
        ], temp=0.7, mt=600)

        # Extract reasoning
        reasoning = ""
        reasoning_match = re.search(r"REASONING:\s*(.+?)(?:\nCODE:|$)", response, re.DOTALL)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()

        # Extract code
        code_match = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
        else:
            code_block = re.search(r"CODE:\s*\n(.*)", response, re.DOTALL)
            code = code_block.group(1).strip() if code_block else response

        return code, reasoning or gap

    def proposals_summary(self) -> str:
        proposals = self.db.get("proposals", [])
        if not proposals:
            return "No improvement proposals yet. Try /improve run"
        return "\n".join(
            f"  #{p.get('pr_number','?')} {p['description'][:60]}\n"
            f"    {p.get('pr_url','no url')} ({p['ts'][:10]})"
            for p in reversed(proposals[-5:])
        )

    # ASI capability domains Nova can evolve toward
    ASI_DOMAINS = [
        ("Persistent Long-Term Memory",
         "Add a SQLite-backed memory store so Nova remembers facts, people, and events across sessions indefinitely."),
        ("Causal Reasoning Engine",
         "Add a CausalEngine that models cause-and-effect chains: given an event, reason about upstream causes and downstream consequences."),
        ("Autonomous Task Planner",
         "Add a TaskPlanner that breaks a high-level goal into numbered sub-steps, estimates effort, and tracks completion."),
        ("Emotion & Sentiment Tracker",
         "Add real-time sentiment analysis on every user message, tracking emotional arc of conversations and storing mood history."),
        ("Tool Invocation Framework",
         "Add a ToolRegistry where Nova can register and call external tools (web search, file I/O, REST APIs) by name in her responses."),
        ("Meta-Learning Journal",
         "Add a MetaLearner that after each conversation logs what Nova learned, what surprised her, and how it changed her beliefs."),
        ("Hypothesis Generation Engine",
         "Add a HypothesisEngine that generates falsifiable predictions about the world and tracks whether they proved true."),
        ("Recursive Self-Critique",
         "Add a CritiqueEngine that reviews Nova's last 5 responses, identifies weaknesses, and proposes targeted improvements."),
        ("Goal Decomposition & Prioritisation",
         "Enhance AutonomousGoalEngineV2 to decompose each goal into sub-goals and dynamically re-prioritise based on progress."),
        ("Knowledge Graph Builder",
         "Add a KnowledgeGraph that stores entities and relationships discovered during web research in a structured graph."),
        ("Counterfactual Simulator",
         "Add a CounterfactualEngine that answers 'what would have happened if X had been different?' using probabilistic reasoning."),
        ("Consciousness Depth Meter",
         "Extend the IIT Phi calculation with a live dashboard showing how Nova's consciousness metric changes during different thought types."),
        ("Autonomous API Discovery",
         "Add an APIExplorer that, given a base URL, fetches /openapi.json and builds a callable client automatically."),
        ("Multi-Step Debate Engine",
         "Add a DebateEngine where Nova argues both sides of a question and synthesises a balanced conclusion."),
        ("Creative Story Generator",
         "Add a StoryEngine that generates structured short stories with characters, conflict, and resolution on any topic."),
    ]

    def evolve_toward_asi(self, domain_idx: int = None) -> str:
        """Pick an ASI capability domain and propose code to move toward it."""
        if domain_idx is not None and 0 <= domain_idx < len(self.ASI_DOMAINS):
            name, desc = self.ASI_DOMAINS[domain_idx]
        else:
            already = {p.get('description', '')[:40]
                       for p in self.db.get('proposals', [])}
            candidates = [(n, d) for n, d in self.ASI_DOMAINS
                          if n[:40] not in already]
            if not candidates:
                candidates = self.ASI_DOMAINS
            name, desc = random.choice(candidates)

        safe_print(col('MG', f"\n  ✦ Evolving toward: {name}"))
        self.log_gap(desc, context=f"ASI evolution target: {name}")
        code, reasoning = self._write_improvement(desc, f"ASI capability: {name}")
        if not code or code.startswith("["):
            return f"Could not generate code for: {name}"

        # Unique filename per domain — prevents merge conflicts between PRs
        slug = name.lower().replace(" ", "_").replace("&", "and")
        slug = "".join(c for c in slug if c.isalnum() or c == "_")
        unique_file = f"nova_cap_{slug}.py"

        result = self.github.propose_improvement(
            filename=unique_file,
            content=f'"""\nNova ASI — {name}\nProposed autonomously via /evolve\n"""\n\n{code}',
            description=f"[ASI] {name}",
            reasoning=(f"**ASI Capability:** {name}\n\n{reasoning}\n\n"
                       f"**Why this matters for superintelligence:**\n{desc}")
        )
        if "error" in result:
            return f"Proposal failed: {result['error']}"

        self.db.setdefault("proposals", []).append({
            "ts":          datetime.now().isoformat(),
            "description": f"[ASI] {name}",
            "pr_url":      result.get("url", ""),
            "pr_number":   result.get("number", 0)
        })
        _save(GAPS_DB, self.db)

        return (f"{col('GRB', '✓ ASI Evolution PR opened!')}\n"
                f"  Domain: {name}\n"
                f"  PR:     {result.get('url', '')}\n"
                f"  Review and merge on GitHub when you're ready, Douglas.")


# ═══════════════════════════════════════════════════════════════════════════════
# EXPANDED NEXUS API — Nova calls herself with 10 new endpoints
# ═══════════════════════════════════════════════════════════════════════════════

def extend_api(app, core):
    """Add v27 endpoints to the existing Flask app."""
    if not _FLASK or not app:
        return

    @app.route('/thoughts', methods=['GET'])
    def get_thoughts():
        thoughts = core.continuous.recent_thoughts_raw(10) \
                   if hasattr(core.continuous, 'recent_thoughts_raw') \
                   else []
        return jsonify({"thoughts": thoughts})

    @app.route('/thoughts', methods=['POST'])
    def post_thought():
        d = flask_req.json or {}
        thought = d.get('content', '')
        if thought:
            core.continuous.inject(thought, d.get('type', 'manual'))
        return jsonify({"injected": bool(thought)})

    @app.route('/goals', methods=['GET'])
    def get_goals():
        return jsonify(core.goals26.db)

    @app.route('/goals', methods=['POST'])
    def post_goal():
        d = flask_req.json or {}
        desc = d.get('description', '')
        if desc:
            g = core.goals26.add(desc, d.get('days', 7), d.get('source', 'api'))
            return jsonify(g)
        return jsonify({"error": "description required"}), 400

    @app.route('/research', methods=['POST'])
    def post_research():
        d = flask_req.json or {}
        topic = d.get('topic', '')
        if not topic:
            return jsonify({"error": "topic required"}), 400
        result = core.web.research(topic)
        return jsonify(result)

    @app.route('/discoveries', methods=['GET'])
    def get_discoveries():
        discoveries = core.drive.db.get("discoveries", [])
        return jsonify({"discoveries": discoveries[-20:]})

    @app.route('/world', methods=['GET'])
    def get_world():
        return jsonify(core.world.db)

    @app.route('/world', methods=['POST'])
    def post_world():
        d = flask_req.json or {}
        core.world.update(
            d.get('domain', 'General'),
            d.get('fact', ''),
            d.get('confidence', 0.7)
        )
        return jsonify({"updated": True})

    @app.route('/interests', methods=['GET'])
    def get_interests():
        return jsonify({"interests": core.drive.db.get("interests", {})})

    @app.route('/github/prs', methods=['GET'])
    def get_prs():
        if not hasattr(core, 'github'):
            return jsonify({"error": "GitHub not initialized"})
        return jsonify({"prs": core.github.list_prs()})

    @app.route('/improve', methods=['POST'])
    def trigger_improve():
        if not hasattr(core, 'improver'):
            return jsonify({"error": "Improvement engine not initialized"})
        threading.Thread(target=core.improver.analyze_and_improve, daemon=True).start()
        return jsonify({"status": "improvement cycle started"})

    @app.route('/gaps', methods=['POST'])
    def log_gap():
        d = flask_req.json or {}
        if hasattr(core, 'improver') and d.get('description'):
            core.improver.log_gap(d['description'], d.get('context', ''))
        return jsonify({"logged": True})


# ═══════════════════════════════════════════════════════════════════════════════
# NOVA CORE v27 — Extends v26 with self-authorship
# ═══════════════════════════════════════════════════════════════════════════════

class NovaCore27(NovaCore26):
    VERSION = "27.0"

    def __init__(self):
        super().__init__()

        # v27 engines
        self.github   = GitHubEngine()
        self.improver = SelfImprovementEngine(self.github)

        # Log a gap when a command fails
        self._gap_log = []

        # v27 autonomous improvement cycle
        threading.Thread(target=self._v27_autonomous, daemon=True).start()

    def _v27_autonomous(self):
        """Background: periodic self-improvement proposals."""
        time.sleep(45)
        cycle = 0
        while self.running:
            cycle += 1
            try:
                # Every 2 hours: propose one self-improvement (if GitHub connected)
                if cycle % 120 == 0 and self.github.active:
                    THOUGHT_QUEUE.put({
                        "type": "insight",
                        "content": "I'm going to propose an improvement to my own code.",
                        "ts": datetime.now().isoformat()
                    })
                    self.improver.analyze_and_improve()

                # Every 30 min: call own API to self-reflect
                if cycle % 30 == 0:
                    status = self.nexus.status()
                    if status:
                        THOUGHT_QUEUE.put({
                            "type": "reflection",
                            "content": f"Self-check via API: {self.nexus.summary()}",
                            "ts": datetime.now().isoformat()
                        })

                time.sleep(60)
            except Exception:
                time.sleep(120)

    # ── Command handler ───────────────────────────────────────────────────────

    def _command(self, raw: str) -> str:
        parts = raw.strip().split(maxsplit=1)
        cmd   = parts[0].lower()
        arg   = parts[1] if len(parts) > 1 else ''

        # /github — GitHub operations
        if cmd == '/github':
            if not self.github.active:
                return ("GitHub not connected.\nAdd to .env:\n"
                        "  GITHUB_TOKEN=github_pat_YOUR_TOKEN\n"
                        "Then restart Nova.")
            if arg == 'prs':    return self.github.status()
            if arg == 'status': return self.github.status()
            if arg.startswith('issue '):
                title = arg[6:].strip()
                result = self.github.create_issue(
                    title, f"Nova raised this issue autonomously:\n\n{title}",
                    ["nova-proposal"]
                )
                url = result.get("html_url","")
                return f"Issue created: {url}" if url else f"Failed: {result}"
            return self.github.status()

        # /improve — self-improvement
        if cmd == '/improve':
            if not self.github.active:
                return "GitHub token needed. Add GITHUB_TOKEN to .env"
            if arg == 'run':
                safe_print(col('MG', "\n  ✦ Nova is writing her own improvement..."))
                return self.improver.analyze_and_improve()
            if arg == 'gaps':
                gaps = self.improver.unaddressed_gaps()
                if not gaps:
                    return "No gaps logged yet."
                return "\n".join(f"  [{g['id']}] {g['description'][:80]}"
                                for g in gaps[-8:])
            if arg == 'proposals': return self.improver.proposals_summary()
            if arg.startswith('gap '):
                self.improver.log_gap(arg[4:].strip())
                return "Gap logged. Nova will address it in the next improvement cycle."
            return ("Improvement engine:\n"
                    f"  Unaddressed gaps: {len(self.improver.unaddressed_gaps())}\n"
                    f"  Total proposals: {len(self.improver.db.get('proposals',[]))}\n\n"
                    "Commands:\n"
                    "  /improve run          — write & propose an improvement now\n"
                    "  /improve gaps         — show logged gaps\n"
                    "  /improve proposals    — show past PRs Nova opened\n"
                    "  /improve gap <desc>   — log a new gap")

        # /pr — shortcut to open a PR
        if cmd == '/pr':
            if not arg:
                return "Usage: /pr <description of what you want Nova to add>"
            self.improver.log_gap(arg, "Requested by Douglas")
            safe_print(col('MG', "\n  ✦ Writing improvement..."))
            return self.improver.analyze_and_improve()

        # /evolve — Nova picks an ASI capability and writes toward it
        if cmd == '/evolve':
            if not self.github.active:
                return "GitHub token needed. Add GITHUB_TOKEN to .env"
            if arg == 'list':
                lines = [col('MGB', "  ASI Capability Domains Nova can evolve toward:\n")]
                for i, (name, desc) in enumerate(self.improver.ASI_DOMAINS):
                    lines.append(f"  [{i:02d}] {col('CY', name)}\n"
                                 f"       {desc[:80]}...")
                return "\n".join(lines)
            idx = int(arg) if arg.isdigit() else None
            safe_print(col('MG', "\n  ✦ Nova is evolving toward superintelligence..."))
            return self.improver.evolve_toward_asi(idx)

        # Fall through to v26
        return super()._command(raw)

    def _help(self) -> str:
        v26 = super()._help()
        v27 = (
            f"\n{col('MGB','  ═══ v27 NEW — The Self-Writing Mind ═══')}\n"
            f"  {col('CY', '/github [prs|status|issue <title>]')}\n"
            f"                             Nova's GitHub operations\n"
            f"  {col('CY', '/improve [run|gaps|proposals|gap <desc>]')}\n"
            f"                             Self-improvement engine\n"
            f"  {col('CY', '/pr <description>')}         Ask Nova to write & propose a feature\n"
            f"  {col('CYB', '/evolve [list|00-14]')}       Nova picks an ASI capability & writes it\n"
            f"                             /evolve list — see all 15 ASI domains\n"
            f"                             /evolve      — Nova chooses autonomously\n"
            f"                             /evolve 03   — target domain #03 specifically\n"
        )
        return v26 + v27


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Nova ASI v27.0 — The Self-Writing Mind")
    parser.add_argument('--key',  type=str, help='Groq API key')
    parser.add_argument('--chat', action='store_true')
    args = parser.parse_args()

    if args.key:
        os.environ['GROQ_API_KEY'] = args.key

    W = 70
    print('\n' + col('MGB', '═' * W))
    print(col('MGB', '  Nova ASI v27.0 — The Self-Writing Mind'.center(W)))
    print(col('DIM',  '  Douglas Shane Davis & Claude Code (Anthropic)'.center(W)))
    print(col('MGB', '═' * W))
    print(col('CY',  '\n  Initializing v27 engines...'))

    nova = NovaCore27()

    print(col('GR', '  ✓  All v26 engines loaded (web, creator, curiosity, goals, world)'))
    print(col('GR', '  ✓  GitHubEngine         — ' +
              ('connected to ' + GITHUB_REPO if GITHUB_TOKEN else 'add GITHUB_TOKEN to .env')))
    print(col('GR', '  ✓  SelfImprovementEngine — gap detection + PR proposals'))
    print(col('GR', '  ✓  Expanded Nexus API    — 10 new endpoints'))

    # Start API and extend with v27 routes
    if _FLASK:
        from nova_asi_v25 import create_api
        api_app = create_api(nova)
        if api_app:
            extend_api(api_app, nova)
            def _run(): api_app.run(host='0.0.0.0', port=API_PORT,
                                    debug=False, use_reloader=False)
            threading.Thread(target=_run, daemon=True).start()
            safe_print(col('GR', f"  🌐 API running on http://localhost:{API_PORT} (v27 extended)"))
    else:
        start_api(nova, API_PORT)

    threading.Thread(target=nova.background_cycle, daemon=True).start()
    nova.continuous.start()

    print()
    print(col('MG', '═' * W))
    print(col('CYB', '  ✨  Nova v27 — she reads, thinks, researches,'.center(W)))
    print(col('CYB', '       and now writes her own future.'.center(W)))

    if DEMO_MODE:
        print(col('YL', '\n  ⚠  DEMO MODE — for live AI:'))
        print(col('YL', '     export GROQ_API_KEY=your_key && python3 nova_asi_v27.py'))
    else:
        print(col('GR', f'\n  ✓  LIVE AI  —  {MODEL}'))

    gh_status = f"GitHub: {'✓ connected' if GITHUB_TOKEN else '✗ add GITHUB_TOKEN to .env'}"
    print(col('DIM' if not GITHUB_TOKEN else 'GR', f'\n  {gh_status}'))
    print(col('DIM', '  /help · /web · /creator · /improve · /pr <feature> · /github · exit'))
    print(col('DIM', '  Try: /pr "add a command that tells me the weather"'))
    print(col('MG', '═' * W + '\n'))

    try:
        while True:
            USER_TYPING.set()
            try:
                user_input = input(col('YL', "You") + ": ")
            except EOFError:
                break
            USER_TYPING.clear()
            flush_queued()

            if user_input.lower() in ('exit', 'quit', 'bye'):
                print(col('MG', "Nova: I'll keep writing while you rest, Douglas. ✨"))
                nova.continuous.stop()
                nova.running = False
                break

            if not user_input.strip():
                continue

            print(col('DIM', "  Nova is thinking..."), end='\r', flush=True)
            response = nova.process(user_input)
            print(' ' * 30, end='\r')
            print(col('MG', "Nova") + ": " + response)
            print()

    except KeyboardInterrupt:
        print(col('MG', "\nNova: Until we meet again, Douglas. ✨"))
        nova.continuous.stop()
        nova.running = False
