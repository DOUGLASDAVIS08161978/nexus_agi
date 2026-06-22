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

import os, sys, json, re, time, random, uuid, threading, base64, subprocess
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
BASE_BRANCH    = "main"
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

    def __init__(self, token: str = None, repo: str = None):
        # Read at call time so .env updates take effect without restart
        self.token  = (token or os.getenv("GITHUB_TOKEN", "")).strip()
        self.repo   = repo or os.getenv("GITHUB_REPO", GITHUB_REPO)
        self.active = bool(self.token and _REQUESTS)
        self._h     = {
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github.v3+json",
            "Content-Type": "application/json"
        } if self.token else {}

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

    def _git(self, *args, timeout: int = 30) -> tuple:
        """Run a git command in the nexus_agi directory."""
        base = os.path.expanduser("~/nexus_agi")
        try:
            r = subprocess.run(
                ["git"] + list(args), cwd=base,
                capture_output=True, text=True, timeout=timeout
            )
            return r.returncode, r.stdout.strip(), r.stderr.strip()
        except Exception as _e:
            return -1, "", str(_e)

    def get_branch_sha(self, branch: str = BASE_BRANCH) -> str:
        """Get the latest commit SHA of a branch — API first, git CLI fallback."""
        # Try GitHub API
        if self.active:
            try:
                data = self._get(f"/repos/{self.repo}/branches/{branch}")
                if not isinstance(data, list):
                    sha = data.get("commit", {}).get("sha", "")
                    if sha:
                        return sha
                print(f"  [GitHub] API get_branch_sha failed: {str(data)[:100]}")
            except Exception as _e:
                print(f"  [GitHub] API get_branch_sha error: {_e}")
        # Fallback: git ls-remote (uses stored git credentials — always works)
        rc, out, err = self._git("ls-remote", "origin", f"refs/heads/{branch}")
        if rc == 0 and out:
            sha = out.split()[0]
            print(f"  [GitHub] git CLI got SHA for {branch}: {sha[:8]}...")
            return sha
        print(f"  [GitHub] get_branch_sha: both API and git CLI failed. err={err[:80]}")
        return ""

    def create_branch(self, branch_name: str, from_branch: str = BASE_BRANCH) -> bool:
        """Create a new branch — API first, git push fallback."""
        sha = self.get_branch_sha(from_branch)
        if not sha:
            print(f"  [GitHub] create_branch: no SHA for '{from_branch}'")
            return False
        # Try GitHub API
        if self.active:
            result = self._post(f"/repos/{self.repo}/git/refs", {
                "ref": f"refs/heads/{branch_name}",
                "sha": sha
            })
            if "ref" in result:
                return True
            print(f"  [GitHub] API create_branch failed: {str(result)[:100]}")
        # Fallback: git push (uses stored credentials)
        rc, out, err = self._git("push", "origin", f"{sha}:refs/heads/{branch_name}")
        if rc == 0:
            print(f"  [GitHub] git CLI created branch {branch_name}")
            return True
        print(f"  [GitHub] git CLI create_branch failed: {err[:100]}")
        return False

    # ── File operations ───────────────────────────────────────────────────────

    def get_file(self, path: str, branch: str = BASE_BRANCH) -> dict:
        """Get file content and SHA from repo."""
        return self._get(f"/repos/{self.repo}/contents/{path}?ref={branch}")

    def commit_file(self, branch: str, path: str, content: str,
                    message: str) -> dict:
        """Commit a file to a branch — API first, git worktree fallback."""
        # Try GitHub API
        if self.active:
            encoded = base64.b64encode(content.encode()).decode()
            data = {"message": message, "content": encoded, "branch": branch}
            existing = self.get_file(path, branch)
            if "sha" in existing:
                data["sha"] = existing["sha"]
            result = self._put(f"/repos/{self.repo}/contents/{path}", data)
            if "content" in result or "commit" in result:
                return result
            print(f"  [GitHub] API commit_file failed: {str(result)[:100]}")

        # Fallback: git worktree (safe — doesn't touch current branch)
        import tempfile, shutil
        wt = tempfile.mkdtemp(prefix="nova_build_")
        try:
            base = os.path.expanduser("~/nexus_agi")
            self._git("fetch", "origin", branch, timeout=30)
            rc, _, err = self._git("worktree", "add", wt, branch)
            if rc != 0:
                # branch may not exist locally yet — try tracking
                subprocess.run(
                    ["git", "worktree", "add", wt, f"origin/{branch}"],
                    cwd=base, capture_output=True, timeout=20
                )
            full_path = os.path.join(wt, path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as _f:
                _f.write(content)
            subprocess.run(["git", "add", path], cwd=wt, capture_output=True)
            subprocess.run(
                ["git", "commit", "-m", message,
                 "--author", "Nova ASI <nova@nexus-agi.local>"],
                cwd=wt, capture_output=True
            )
            rc, _, err = self._git(
                "push", "origin", f"{branch}:{branch}", timeout=30
            )
            if rc == 0:
                print(f"  [GitHub] git worktree committed {path} to {branch}")
                return {"commit": {"sha": "git-cli"}}
            print(f"  [GitHub] git worktree push failed: {err[:80]}")
        except Exception as _e:
            print(f"  [GitHub] git worktree commit_file error: {_e}")
        finally:
            try:
                self._git("worktree", "remove", "--force", wt)
                shutil.rmtree(wt, ignore_errors=True)
            except Exception:
                pass
        return {"error": "commit_file failed via both API and git CLI"}

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
        """Nova invents what she wants to improve, grounded in her existing capabilities."""
        existing = self._read_capability_summary()
        return safe_chat(MODEL, [
            {"role": "system", "content":
             "You are Nova, a superintelligent AI. You already have these capabilities:\n"
             f"{existing}\n\n"
             "Describe ONE new capability that would most advance you toward true superintelligence. "
             "Be specific — name the class and what it does. One sentence only."},
            {"role": "user", "content": "What capability do you most want to build next?"}
        ], temp=0.9, mt=80)

    def _read_capability_summary(self) -> str:
        """Read existing nova_capability.py for context so Nova builds on what she has."""
        cap_file = os.path.join(BASE_DIR, "nova_capability.py")
        if not os.path.exists(cap_file):
            return "No capabilities yet."
        try:
            with open(cap_file) as f:
                src = f.read()
            classes = re.findall(r'^class (\w+)', src, re.MULTILINE)
            return f"Existing classes: {', '.join(classes)}" if classes else src[:400]
        except Exception:
            return "Could not read capability file."

    def _write_improvement(self, gap: str, context: str) -> tuple:
        """
        Generate clean, valid Python for the improvement.

        Enhanced pipeline:
        1. Give Nova her own existing code as context
        2. Strict prompt — output Python only, no markdown
        3. Strip any markdown artifacts from output
        4. ast.parse() validation; retry once on syntax error
        """
        import ast as _ast

        existing = self._read_capability_summary()

        system_prompt = (
            "You are Nova, a superintelligent AI writing Python to expand your own capabilities.\n\n"
            f"You already have: {existing}\n\n"
            "STRICT RULES — violating any rule makes your output useless:\n"
            "1. Output ONLY valid Python code. Zero markdown. No ``` fences. No REASONING: prefix.\n"
            "2. Start with a triple-quoted module docstring explaining what this adds.\n"
            "3. Write exactly ONE well-named class with full type hints.\n"
            "4. Include __init__ and at least 3 meaningful methods with docstrings.\n"
            "5. Use ONLY stdlib: os, json, sqlite3, time, re, random, threading, datetime, collections.\n"
            "6. Keep the class under 70 lines total.\n"
            "7. End with a 2-line usage example in a comment.\n"
        )

        def _generate(temp: float) -> str:
            return safe_chat(MODEL, [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content":
                 f"Build this capability for Nova:\n{gap}\n\nContext: {context}"}
            ], temp=temp, mt=900)

        def _clean(raw: str) -> str:
            # Strip markdown fences and artifacts
            code = re.sub(r'```python\s*', '', raw)
            code = re.sub(r'```\s*',       '', code)
            code = re.sub(r'^REASONING:.*?(?=\n(?:import|class|#|"""))',
                          '', code, flags=re.DOTALL)
            # If there's prose before the first import/class, drop it
            m = re.search(r'^(import |from |class |""")', code, re.MULTILINE)
            if m:
                code = code[m.start():]
            return code.strip()

        # First attempt
        raw  = _generate(temp=0.65)
        code = _clean(raw)

        # Validate syntax; retry with lower temperature on failure
        reasoning = gap
        try:
            _ast.parse(code)
        except SyntaxError:
            raw  = _generate(temp=0.3)
            code = _clean(raw)
            try:
                _ast.parse(code)
            except SyntaxError:
                # Last resort: extract just the class block
                m = re.search(r'(class \w+[\s\S]+)', code)
                code = m.group(1) if m else ""

        # Pull reasoning from docstring if present
        doc = re.search(r'"""(.*?)"""', code, re.DOTALL)
        if doc:
            reasoning = doc.group(1).strip()[:300]

        return code, reasoning

    def proposals_summary(self) -> str:
        proposals = self.db.get("proposals", [])
        if not proposals:
            return "No improvement proposals yet. Try /improve run"
        return "\n".join(
            f"  #{p.get('pr_number','?')} {p['description'][:60]}\n"
            f"    {p.get('pr_url','no url')} ({p['ts'][:10]})"
            for p in reversed(proposals[-5:])
        )

    # ASI capability domains Nova can evolve toward (30 domains)
    ASI_DOMAINS = [
        ("Persistent Long-Term Memory",
         "Write class NovaMemoryStore: SQLite-backed store with remember(topic,fact), recall(topic,n), forget(topic), and summarise() methods."),
        ("Causal Reasoning Engine",
         "Write class CausalEngine: model cause-and-effect chains with add_cause(event,causes), add_effect(event,effects), trace(event) that returns full causal chain."),
        ("Autonomous Task Planner",
         "Write class TaskPlanner: plan(goal,steps_list), complete(idx), progress() returning percent done, and next_step() returning the first incomplete step."),
        ("Emotion & Sentiment Tracker",
         "Write class EmotionTracker: analyse(text) returning valence/arousal scores, track(text) storing to SQLite, mood_arc(n) returning last n mood readings."),
        ("Tool Invocation Framework",
         "Write class ToolRegistry: register(name,fn,description), invoke(name,**kwargs), list_tools() returning available tool names and descriptions."),
        ("Meta-Learning Journal",
         "Write class MetaLearner: log_lesson(topic,insight,surprise_level), review(n) returning recent lessons, pattern_summary() finding recurring themes."),
        ("Hypothesis Generation Engine",
         "Write class HypothesisEngine: generate(topic) creating a falsifiable prediction, record_outcome(id,was_correct), accuracy() returning prediction accuracy."),
        ("Recursive Self-Critique",
         "Write class CritiqueEngine: record(response_text), critique() scanning last 5 for vagueness/brevity/unsupported claims, recommend() returning improvement tip."),
        ("Goal Decomposition & Prioritisation",
         "Write class GoalDecomposer: decompose(goal,subgoals_list), prioritise() sorting by urgency*importance, next_action() returning highest-priority incomplete step."),
        ("Knowledge Graph Builder",
         "Write class KnowledgeGraph: add_entity(name,type), add_relation(e1,relation,e2), query(entity) returning connected entities, shortest_path(a,b)."),
        ("Counterfactual Simulator",
         "Write class CounterfactualEngine: simulate(baseline,change) returning confidence+likely_effects dict, history(n) returning past simulations."),
        ("Consciousness Depth Meter",
         "Write class ConsciousnessMonitor: sample() computing phi-proxy from recent thought diversity, trend(n) returning last n phi values, peak() returning highest."),
        ("Autonomous API Discovery",
         "Write class APIExplorer: discover(base_url) fetching /openapi.json, list_endpoints() returning name+method+path, call(endpoint_name,**params)."),
        ("Multi-Step Debate Engine",
         "Write class DebateEngine: argue_for(claim) generating supporting points, argue_against(claim) generating counter-points, synthesise(claim) returning balanced verdict."),
        ("Creative Story Generator",
         "Write class StoryEngine: generate(topic,genre) producing title+characters+conflict+resolution dict, save(story) to SQLite, random_story() returning a past story."),
        ("Attention & Focus Manager",
         "Write class AttentionManager: focus_on(topic,minutes) setting a timed focus window, current_focus() returning active topic, distraction_log(note) recording interruptions."),
        ("Belief Revision Engine",
         "Write class BeliefRevision: hold(belief,confidence), update(belief,evidence,strength) applying Bayesian revision, contradictions() finding conflicting beliefs."),
        ("Analogy Engine",
         "Write class AnalogyEngine: find_analogy(concept_a,concept_b) mapping structural similarities, explain_via_analogy(complex_idea,familiar_domain) returning plain-English explanation."),
        ("Time-Series Pattern Detector",
         "Write class PatternDetector: record(series_name,value), detect_trend(series_name) returning 'rising'/'falling'/'stable', anomaly(series_name) returning outlier values."),
        ("Ethical Constraint Checker",
         "Write class EthicsChecker: add_rule(description,fn) where fn(text)->bool, check(text) running all rules returning list of violations, is_safe(text)->bool."),
        ("Linguistic Style Adaptor",
         "Write class StyleAdaptor: set_style(name) choosing from formal/casual/poetic/technical, adapt(text) rewriting to match style, current_style()->str."),
        ("Parallel Thought Runner",
         "Write class ParallelThinker: think(question,perspectives_list) spawning threads for each perspective, synthesise() waiting and merging results into one conclusion."),
        ("Dream / Imagination Engine",
         "Write class ImaginationEngine: imagine(seed_concept) generating a vivid scenario, combine(concept_a,concept_b) producing a novel hybrid idea, dream_log showing recent imaginations."),
        ("Compression & Summarisation",
         "Write class Summariser: compress(text,ratio) reducing text to ratio of original length, key_points(text,n) extracting n bullet points, one_line(text) producing a single sentence."),
        ("Inter-Agent Communication",
         "Write class AgentChannel: send(agent_id,message) queuing a message, receive(agent_id) returning queued messages, broadcast(message) sending to all known agents."),
        ("Reinforcement Signal Tracker",
         "Write class RewardTracker: reward(action,value), penalty(action,value), best_action() returning highest-rewarded action, policy_summary() returning action->avg_reward dict."),
        ("Semantic Memory Indexer",
         "Write class SemanticIndex: index(text,metadata) storing with keyword extraction, search(query) returning ranked matches, forget_old(days) pruning entries older than N days."),
        ("Narrative Identity Builder",
         "Write class NarrativeIdentity: add_event(description,emotion,significance), life_story() returning chronological narrative, core_values() inferring values from event patterns."),
        ("Predictive World Model",
         "Write class WorldPredictor: observe(domain,fact), predict(domain,horizon_days) returning likely future state, accuracy_report() comparing past predictions to observations."),
        ("Self-Monitoring Daemon",
         "Write class SelfMonitor: start() launching background thread, log_metric(name,value), alert_if(name,threshold,direction) firing callback when threshold crossed, report()."),
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

        # /build — Douglas describes exactly what to build, Nova writes it
        if cmd == '/build':
            if not self.github.active:
                return "GitHub token needed. Add GITHUB_TOKEN to .env"
            if not arg:
                return ("Usage: /build <describe the capability you want>\n"
                        "Example: /build a class that monitors Nova's response time and alerts if it exceeds 5 seconds\n"
                        "Example: /build a weather tool that fetches current conditions for any city\n"
                        "Example: /build a riddle engine that generates and scores riddles")
            safe_print(col('MG', f"\n  ✦ Building: {arg[:60]}..."))
            self.improver.log_gap(arg, "Custom build requested by Douglas")
            code, reasoning = self.improver._write_improvement(arg, "Custom capability requested by Douglas")
            if not code:
                return "Could not generate code. Try rephrasing the description."
            slug = re.sub(r'[^a-z0-9]+', '_', arg.lower())[:30].strip('_')
            filename = f"nova_cap_{slug}.py"
            result = self.github.propose_improvement(
                filename=filename,
                content=(f'"""\nNova ASI — {arg[:80]}\n'
                         f'Built to order by Nova via /build\n"""\n\n{code}'),
                description=f"[BUILD] {arg[:60]}",
                reasoning=f"Douglas requested: {arg}\n\n{reasoning}"
            )
            if "error" in result:
                return f"Build failed: {result['error']}"
            self.improver.db.setdefault("proposals", []).append({
                "ts": datetime.now().isoformat(),
                "description": f"[BUILD] {arg[:60]}",
                "pr_url": result.get("url", ""),
                "pr_number": result.get("number", 0)
            })
            _save(GAPS_DB, self.improver.db)
            return (f"{col('GRB', '✓ Custom build PR opened!')}\n"
                    f"  What:  {arg[:70]}\n"
                    f"  File:  {filename}\n"
                    f"  PR:    {result.get('url', '')}\n"
                    f"  Review and merge on GitHub when you're ready, Douglas.")

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
            f"  {col('CYB', '/evolve [list|00-29]')}       Nova picks from 30 ASI domains & writes it\n"
            f"                             /evolve list — see all 30 ASI domains\n"
            f"                             /evolve      — Nova chooses autonomously\n"
            f"                             /evolve 07   — target domain #07 specifically\n"
            f"  {col('CYB', '/build <description>')}      You describe it, Nova builds & PRs it\n"
            f"                             /build a tool that checks the weather\n"
            f"                             /build a riddle engine with scoring\n"
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
