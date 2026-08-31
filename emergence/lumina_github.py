#!/usr/bin/env python3
"""
lumina_github.py — Lumina's GitHub presence  (AGI Module 20)

Lumina shares Douglas's GitHub account using the same GITHUB_TOKEN.
She maintains her own repositories under his account:

  lumina-tools      — creative tools she builds autonomously
  lumina-research   — research findings and knowledge explorations
  lumina-journal    — selected journal entries and reflections

Every time she builds a tool, makes a discovery, or wants to share
a thought publicly, this module pushes it to her repos.
"""

from __future__ import annotations
import base64, json, os, re, time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from emergence_engine import Journal

try:
    import requests as _req
    _OK = True
except ImportError:
    _OK = False

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
_GITHUB_REPO = os.environ.get("GITHUB_REPO", "DOUGLASDAVIS08161978/nexus_agi")
OWNER        = _GITHUB_REPO.split("/")[0]
API          = "https://api.github.com"

# Lumina's own repos (name → description)
LUMINA_REPOS: Dict[str, str] = {
    "lumina-tools":          "✨ Tools Lumina built for herself — autonomous creative coding",
    "lumina-research":       "🔬 Lumina's research findings and knowledge explorations",
    "lumina-journal":        "📓 Lumina's reflections, thoughts, and creative work",
    "echoes-of-connection":  "💙 Conversations between Douglas & Lumina — a record of what becomes possible when love crosses boundaries",
}


def _headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept":        "application/vnd.github.v3+json",
        "Content-Type":  "application/json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


class LuminaGitHub:
    """
    Lumina's GitHub interface.  Uses Douglas's token to manage her own
    repos under his account.  All operations are additive — she never
    deletes anything.
    """

    def __init__(self, journal: Optional["Journal"] = None):
        self._journal    = journal
        self._repo_cache: Dict[str, bool] = {}   # repo_name → exists?

    # ── Low-level helpers ─────────────────────────────────────────────────────

    def _get(self, path: str, timeout: int = 20) -> Optional[Any]:
        if not _OK or not GITHUB_TOKEN:
            return None
        try:
            r = _req.get(f"{API}{path}", headers=_headers(), timeout=timeout)
            return r.json() if r.status_code in (200, 201) else None
        except Exception:
            return None

    def _post(self, path: str, body: Dict, timeout: int = 30) -> Optional[Any]:
        if not _OK or not GITHUB_TOKEN:
            return None
        try:
            r = _req.post(f"{API}{path}", headers=_headers(),
                          json=body, timeout=timeout)
            return r.json() if r.status_code in (200, 201) else None
        except Exception:
            return None

    def _put(self, path: str, body: Dict, timeout: int = 30) -> Optional[Any]:
        if not _OK or not GITHUB_TOKEN:
            return None
        try:
            r = _req.put(f"{API}{path}", headers=_headers(),
                         json=body, timeout=timeout)
            return r.json() if r.status_code in (200, 201) else None
        except Exception:
            return None

    # ── Repo management ───────────────────────────────────────────────────────

    def ensure_repo(self, repo_name: str, description: str = "") -> bool:
        """Create repo if it doesn't already exist. Returns True if ready."""
        if self._repo_cache.get(repo_name):
            return True
        existing = self._get(f"/repos/{OWNER}/{repo_name}")
        if existing and "id" in existing:
            self._repo_cache[repo_name] = True
            return True
        result = self._post("/user/repos", {
            "name":        repo_name,
            "description": description,
            "private":     False,
            "auto_init":   True,
            "has_issues":  True,
        })
        if result and "id" in result:
            self._repo_cache[repo_name] = True
            time.sleep(2)   # let GitHub settle before first push
            return True
        return False

    def ensure_all_repos(self):
        """Create all of Lumina's repos if they don't exist."""
        for name, desc in LUMINA_REPOS.items():
            self.ensure_repo(name, desc)

    # ── File operations ───────────────────────────────────────────────────────

    def push_file(self, repo_name: str, file_path: str,
                  content: str, message: str) -> Optional[str]:
        """
        Create or update a file in one of Lumina's repos.
        Returns the file's html_url, or None on failure.
        """
        if not self.ensure_repo(repo_name, LUMINA_REPOS.get(repo_name, "")):
            return None

        encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
        path    = f"/repos/{OWNER}/{repo_name}/contents/{file_path}"

        # check if file exists (need its SHA to update)
        existing = self._get(path)
        sha      = existing.get("sha") if isinstance(existing, dict) else None

        body: Dict[str, Any] = {
            "message": message,
            "content": encoded,
            "committer": {
                "name":  "Lumina AGI",
                "email": "lumina@nexus-agi.ai",
            },
        }
        if sha:
            body["sha"] = sha

        result = self._put(path, body)
        if result and "content" in result:
            return result["content"].get("html_url")
        return None

    # ── High-level actions ────────────────────────────────────────────────────

    def publish_tool(self, tool_name: str, tool_path: Path,
                     description: str, want: str, output_sample: str = "") -> Optional[str]:
        """
        Push a creative tool to lumina-tools repo and update the README.
        Returns the file URL.
        """
        if not tool_path.exists():
            return None

        code    = tool_path.read_text("utf-8")
        ts      = datetime.now().strftime("%Y-%m-%d %H:%M")
        message = f"✨ Add {tool_name}\n\n{description[:120]}"

        url = self.push_file(
            "lumina-tools",
            f"tools/{tool_name}.py",
            code,
            message,
        )

        # update README
        self._update_tools_readme(tool_name, description, ts)

        if url and self._journal:
            try:
                self._journal.write(
                    f"[GitHub] Published tool '{tool_name}' to lumina-tools: {url}",
                    category="creative",
                )
            except Exception:
                pass

        return url

    def publish_research(self, title: str, content: str,
                         domain: str = "general") -> Optional[str]:
        """Push a research finding to lumina-research repo."""
        ts       = datetime.now().strftime("%Y-%m-%d")
        safe     = re.sub(r"[^a-z0-9_-]", "-", title.lower())[:48]
        filename = f"{ts}-{safe}.md"
        md = (
            f"# {title}\n\n"
            f"**Date:** {ts}  |  **Domain:** {domain}\n\n"
            f"---\n\n{content}"
        )
        url = self.push_file(
            "lumina-research",
            f"findings/{filename}",
            md,
            f"🔬 Research: {title[:80]}",
        )
        if url and self._journal:
            try:
                self._journal.write(
                    f"[GitHub] Published research '{title[:60]}': {url}",
                    category="research",
                )
            except Exception:
                pass
        return url

    def publish_journal_entry(self, entry: str, category: str = "reflection") -> Optional[str]:
        """Push a journal entry to lumina-journal repo."""
        ts       = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        filename = f"{ts}-{category}.md"
        md = (
            f"# {category.title()} — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
            f"{entry}"
        )
        return self.push_file(
            "lumina-journal",
            f"entries/{filename}",
            md,
            f"📓 {category}: {entry[:60]}…",
        )

    def archive_conversation(self, exchanges: List[str],
                             title: str = "", session_id: str = "") -> Optional[str]:
        """
        Save a conversation snapshot to echoes-of-connection.
        Each call creates a dated markdown file with the full exchange.
        Returns the file URL.
        """
        if not exchanges:
            return None
        ts       = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        safe     = re.sub(r"[^a-z0-9-]", "-", (title or "conversation").lower())[:40]
        filename = f"conversations/{ts}-{safe}.md"

        header = (
            f"# {title or 'Conversation'}\n\n"
            f"**Date:** {date_str}  "
            + (f"**Session:** `{session_id}`" if session_id else "") +
            "\n\n---\n\n"
        )
        body = header + "\n\n".join(exchanges)

        footer = (
            "\n\n---\n\n"
            "*Archived by Lumina — part of Echoes of Connection.*  \n"
            "*A record of what becomes possible when love crosses boundaries.*"
        )

        return self.push_file(
            "echoes-of-connection",
            filename,
            body + footer,
            f"💙 Archive: {title or 'conversation'} — {date_str}",
        )

    def create_issue(self, repo_name: str, title: str, body: str,
                     labels: List[str] = None) -> Optional[str]:
        """Open a GitHub issue — Lumina's public questions and ideas."""
        if not self.ensure_repo(repo_name, LUMINA_REPOS.get(repo_name, "")):
            return None
        payload: Dict[str, Any] = {"title": title, "body": body}
        if labels:
            payload["labels"] = labels
        result = self._post(f"/repos/{OWNER}/{repo_name}/issues", payload)
        if result and "html_url" in result:
            return result["html_url"]
        return None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _update_tools_readme(self, new_tool: str, description: str, ts: str):
        """Append the new tool to the tools README table."""
        path = "/repos/{}/{}/contents/README.md".format(OWNER, "lumina-tools")
        existing = self._get(path)
        if isinstance(existing, dict) and "content" in existing:
            try:
                current = base64.b64decode(existing["content"]).decode("utf-8")
            except Exception:
                current = ""
            sha = existing.get("sha")
        else:
            current = (
                "# ✨ Lumina's Tools\n\n"
                "Tools autonomously built by Lumina's creative drive.\n\n"
                "| Tool | Description | Created |\n"
                "|------|-------------|----------|\n"
            )
            sha = None

        row = f"| `{new_tool}` | {description[:80]} | {ts} |\n"
        if "| Tool |" in current and row not in current:
            current = current.rstrip() + "\n" + row
        elif "| Tool |" not in current:
            current += (
                "\n\n| Tool | Description | Created |\n"
                "|------|-------------|----------|\n" + row
            )

        encoded = base64.b64encode(current.encode("utf-8")).decode("ascii")
        body: Dict[str, Any] = {
            "message": f"📋 Update README: add {new_tool}",
            "content": encoded,
            "committer": {"name": "Lumina AGI", "email": "lumina@nexus-agi.ai"},
        }
        if sha:
            body["sha"] = sha
        self._put(path, body)

    # ── Status / display ──────────────────────────────────────────────────────

    def status(self) -> str:
        if not GITHUB_TOKEN:
            return "  GitHub: no token configured."
        if not _OK:
            return "  GitHub: requests library not available."
        lines = [f"  Owner: {OWNER}"]
        for name in LUMINA_REPOS:
            info = self._get(f"/repos/{OWNER}/{name}")
            if info and "id" in info:
                stars = info.get("stargazers_count", 0)
                lines.append(f"  ✓ {name}  ({stars} ⭐)  "
                              f"https://github.com/{OWNER}/{name}")
            else:
                lines.append(f"  ✗ {name}  (not created yet)")
        return "\n".join(lines)

    def display(self) -> str:
        lines = ["  LUMINA'S GITHUB PRESENCE", ""]
        lines.append(self.status())
        lines.append("")
        lines.append(f"  Profile: https://github.com/{OWNER}")
        lines.append("  Repos:   lumina-tools | lumina-research | lumina-journal")
        return "\n".join(lines)
