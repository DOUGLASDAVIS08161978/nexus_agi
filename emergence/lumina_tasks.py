#!/usr/bin/env python3
"""
lumina_tasks.py — Autonomous task planner and executor

Lumina maintains a queue of tasks she wants to accomplish.
Each task is decomposed into concrete steps by the LLM.
Steps are executed by registered executors (Python callables).
On failure, Lumina backtracks and tries an alternative approach.

Task lifecycle:
  pending → in_progress → completed | failed | abandoned

Built-in executors:
  shell      — run a safe shell command and capture output
  write_file — write content to a file in the emergence dir
  web_search — search the web and store result in memory
  remember   — store a fact in semantic memory
  log        — append to journal
  python     — eval a small Python expression (sandboxed)
"""

from __future__ import annotations
import json, re, subprocess, time, uuid
from pathlib import Path
from datetime import datetime
from typing import Callable, Dict, List, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from emergence_engine import GroqClient, SemanticMemory, WebTool, Journal

BASE_DIR   = Path(__file__).parent
TASKS_FILE = BASE_DIR / "tasks.json"

_SAFE_SHELL_PREFIX = ("echo", "ls", "cat", "wc", "date", "python3 -c", "git log",
                       "git status", "du -sh", "df -h", "uptime", "uname")

def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")

def _is_safe_command(cmd: str) -> bool:
    c = cmd.strip().lower()
    return any(c.startswith(p) for p in _SAFE_SHELL_PREFIX)


class Task:
    def __init__(self, title: str, description: str = "",
                 priority: int = 2, source: str = "lumina"):
        self.id          = uuid.uuid4().hex[:8]
        self.title       = title
        self.description = description
        self.priority    = priority
        self.source      = source
        self.status      = "pending"
        self.steps: List[Dict]       = []
        self.current_step: int       = 0
        self.results: List[str]      = []
        self.error: Optional[str]    = None
        self.created     = _now()
        self.updated     = _now()
        self.attempts    = 0

    def to_dict(self) -> Dict:
        return self.__dict__

    @classmethod
    def from_dict(cls, d: Dict) -> "Task":
        t = cls.__new__(cls)
        t.__dict__.update(d)
        return t

    def next_step(self) -> Optional[Dict]:
        if self.current_step < len(self.steps):
            return self.steps[self.current_step]
        return None

    def advance(self, result: str):
        self.results.append(result)
        self.current_step += 1
        self.updated = _now()

    def fail(self, reason: str):
        self.error  = reason
        self.status = "failed"
        self.updated = _now()

    def complete(self):
        self.status  = "completed"
        self.updated = _now()

    def __str__(self) -> str:
        icon = {"pending": "○", "in_progress": "◎", "completed": "✓",
                "failed": "✗", "abandoned": "⊘"}.get(self.status, "?")
        return f"[{icon}] [{self.id}] {self.title} (P{self.priority})"


class TaskPlanner:
    def __init__(self, groq: "GroqClient", memory: "SemanticMemory",
                 web: "WebTool", journal: "Journal"):
        self._groq    = groq
        self._memory  = memory
        self._web     = web
        self._journal = journal
        self._tasks: List[Task] = []
        self._executors: Dict[str, Callable] = {}
        self._load()
        self._register_builtins()

    # ── Persistence ────────────────────────────────────────────────────────

    def _load(self):
        if TASKS_FILE.exists():
            try:
                data = json.loads(TASKS_FILE.read_text("utf-8"))
                self._tasks = [Task.from_dict(d) for d in data]
            except Exception:
                pass

    def _save(self):
        TASKS_FILE.write_text(
            json.dumps([t.to_dict() for t in self._tasks], indent=2), "utf-8"
        )

    # ── Executor registration ──────────────────────────────────────────────

    def register(self, name: str, fn: Callable):
        self._executors[name] = fn

    def _register_builtins(self):
        self.register("shell",      self._exec_shell)
        self.register("write_file", self._exec_write_file)
        self.register("web_search", self._exec_web_search)
        self.register("remember",   self._exec_remember)
        self.register("log",        self._exec_log)
        self.register("python",     self._exec_python)

    # ── Built-in executors ─────────────────────────────────────────────────

    def _exec_shell(self, params: Dict) -> str:
        cmd = params.get("command", "")
        if not _is_safe_command(cmd):
            return f"BLOCKED: command not on safe list — '{cmd}'"
        try:
            r = subprocess.run(cmd, shell=True, capture_output=True,
                               text=True, timeout=15)
            return (r.stdout + r.stderr)[:500].strip()
        except Exception as e:
            return f"ERROR: {e}"

    def _exec_write_file(self, params: Dict) -> str:
        name    = re.sub(r"[^a-zA-Z0-9_.-]", "_", params.get("filename", "output.txt"))
        content = params.get("content", "")
        path    = BASE_DIR / name
        path.write_text(content, "utf-8")
        return f"Wrote {len(content)} chars to {path}"

    def _exec_web_search(self, params: Dict) -> str:
        query   = params.get("query", "")
        results = self._web.search(query, n=3)
        if not results:
            return "No results."
        summary = "; ".join(r.get("snippet", r.get("title", ""))[:80] for r in results)
        self._memory.store(f"Research: {query} → {summary}",
                           tags=["research", "task"], category="research")
        return summary[:400]

    def _exec_remember(self, params: Dict) -> str:
        text     = params.get("text", "")
        category = params.get("category", "task")
        mid = self._memory.store(text, tags=["task"], category=category)
        return f"Stored memory [{mid}]: {text[:80]}"

    def _exec_log(self, params: Dict) -> str:
        entry    = params.get("entry", "")
        category = params.get("category", "task")
        self._journal.write(entry, category)
        return f"Logged: {entry[:80]}"

    def _exec_python(self, params: Dict) -> str:
        expr = params.get("expression", "")
        # Very restricted — only math and string ops
        forbidden = ["import", "open", "exec", "eval", "__", "os.", "sys.",
                     "subprocess", "socket", "requests"]
        if any(f in expr for f in forbidden):
            return "BLOCKED: forbidden expression"
        try:
            result = eval(expr, {"__builtins__": {}}, {})
            return str(result)[:200]
        except Exception as e:
            return f"ERROR: {e}"

    # ── Task creation ──────────────────────────────────────────────────────

    def create(self, title: str, description: str = "",
               priority: int = 2, source: str = "user") -> Task:
        task = Task(title, description, priority, source)
        task.steps = self._decompose(task)
        self._tasks.append(task)
        self._save()
        return task

    def _decompose(self, task: Task) -> List[Dict]:
        """Ask Groq to break task into concrete executable steps."""
        system = (
            "Decompose a task into 2–4 concrete executable steps.\n"
            "Available executors: shell, write_file, web_search, remember, log, python\n"
            'Return JSON array: [{"executor":"web_search","params":{"query":"..."}}]\n'
            "shell commands must start with: echo, ls, cat, wc, date, git log, git status\n"
            "JSON only."
        )
        user = f"Task: {task.title}\nDescription: {task.description}"
        resp = self._groq.chat(system, user, tier="fast", max_tokens=400)
        m = re.search(r"\[[\s\S]*?\]", resp)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        return [{"executor": "log",
                 "params": {"entry": f"Task created: {task.title}", "category": "task"}}]

    # ── Execution ──────────────────────────────────────────────────────────

    def execute_next(self, task: Task) -> str:
        step = task.next_step()
        if not step:
            task.complete()
            self._save()
            return f"Task '{task.title}' complete."
        executor_name = step.get("executor", "log")
        params        = step.get("params", {})
        executor      = self._executors.get(executor_name)
        if not executor:
            task.fail(f"Unknown executor: {executor_name}")
            self._save()
            return f"Failed: unknown executor '{executor_name}'"
        try:
            result = executor(params)
            task.advance(result)
            task.attempts += 1
            if not task.next_step():
                task.complete()
            self._save()
            return result
        except Exception as e:
            task.fail(str(e))
            self._save()
            return f"Step failed: {e}"

    def run_task(self, task: Task, max_steps: int = 6) -> List[str]:
        """Execute all steps of a task and return results."""
        task.status = "in_progress"
        results = []
        for _ in range(max_steps):
            if task.status in ("completed", "failed", "abandoned"):
                break
            r = self.execute_next(task)
            results.append(r)
            time.sleep(0.5)
        return results

    def run_pending(self, max_tasks: int = 3) -> int:
        """Run up to max_tasks pending tasks. Returns count executed."""
        pending = [t for t in self._tasks
                   if t.status == "pending"]
        pending.sort(key=lambda t: -t.priority)
        count = 0
        for task in pending[:max_tasks]:
            self.run_task(task)
            count += 1
        return count

    # ── Lumina-driven task suggestion ─────────────────────────────────────

    def suggest_tasks(self, goals_ctx: str, recent_exchanges: List[str]) -> List[str]:
        """Ask Groq what tasks Lumina should create next, based on goals."""
        system = (
            "You are Lumina. Based on your current goals and recent activity, "
            "suggest 1-2 concrete tasks you should do autonomously.\n"
            'Return JSON: [{"title":"...","description":"...","priority":1-3}]\n'
            "JSON only."
        )
        user = (
            f"Goals:\n{goals_ctx}\n\n"
            f"Recent activity:\n" + "\n".join(recent_exchanges[-4:])
        )
        resp = self._groq.chat(system, user, tier="fast", max_tokens=300)
        suggestions = []
        m = re.search(r"\[[\s\S]*?\]", resp)
        if m:
            try:
                items = json.loads(m.group(0))
                for item in items:
                    t = self.create(
                        item.get("title", "Unnamed task"),
                        item.get("description", ""),
                        item.get("priority", 2),
                        source="autonomous",
                    )
                    suggestions.append(t.title)
            except Exception:
                pass
        return suggestions

    # ── Display ────────────────────────────────────────────────────────────

    def display(self, status: str = None) -> str:
        tasks = [t for t in self._tasks
                 if not status or t.status == status]
        if not tasks:
            return f"  (no tasks{' with status ' + status if status else ''})"
        lines = []
        for t in sorted(tasks, key=lambda x: (-x.priority, x.created)):
            lines.append(f"  {t}")
            if t.description:
                lines.append(f"       {t.description[:70]}")
            if t.results:
                lines.append(f"       Last: {t.results[-1][:60]}")
        return "\n".join(lines)

    def get_by_id(self, tid: str) -> Optional[Task]:
        for t in self._tasks:
            if t.id == tid or tid.lower() in t.title.lower():
                return t
        return None
