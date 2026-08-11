import os
import sys
import json
import time
import logging
import subprocess
import requests
import ast
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("lumina_evolution")

class EvolutionPhase(Enum):
    ANALYSIS = "analysis"
    GENERATION = "generation"
    VALIDATION = "validation"
    APPLICATION = "application"
    REFLECTION = "reflection"

@dataclass
class EvolutionConfig:
    groq_api_key: str
    github_token: str
    repo_url: str
    branch_prefix: str = "lumina/evolve"
    max_iterations: int = 5
    dry_run: bool = True
    target_files: Optional[List[str]] = None
    model: str = "llama-3.3-70b-versatile"
    safety_threshold: float = 0.8
    max_diff_size: int = 5000
    cooldown_seconds: int = 30

@dataclass
class EvolutionRecord:
    iteration: int
    phase: str
    timestamp: str
    success: bool
    target_file: str
    change_summary: str
    validation_score: float
    pr_url: Optional[str] = None
    error: Optional[str] = None

@dataclass
class EvolutionState:
    iteration_count: int = 0
    success_rate: float = 0.0
    total_changes: int = 0
    successful_changes: int = 0
    learned_patterns: List[str] = field(default_factory=list)
    recent_records: List[Dict] = field(default_factory=list)
    last_updated: str = ""

class LuminaAutonomousEvolution:
    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.state_file = Path("evolution_state.json")
        self.state = self._load_state()
        self._setup_clients()
        self._parse_repo_info()
        logger.info("Lumina Autonomous Evolution Engine initialized")
        logger.info(f"Config: dry_run={config.dry_run}, max_iterations={config.max_iterations}, model={config.model}")

    def _setup_clients(self):
        try:
            from groq import Groq
            self.groq_client = Groq(api_key=self.config.groq_api_key)
            logger.info("Groq client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}")
            raise RuntimeError("Groq API key invalid or SDK missing")

    def _parse_repo_info(self):
        url = self.config.repo_url.rstrip("/")
        if url.startswith("https://github.com/"):
            parts = url.split("/")
            self.repo_owner = parts[3]
            self.repo_name = parts[4].replace(".git", "")
        elif url.startswith("git@github.com:"):
            parts = url.split(":")[1].split("/")
            self.repo_owner = parts[0]
            self.repo_name = parts[1].replace(".git", "")
        else:
            raise ValueError("Unsupported repository URL format. Use GitHub HTTPS or SSH format.")
        logger.info(f"Target repository: {self.repo_owner}/{self.repo_name}")

    def _load_state(self) -> EvolutionState:
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    data = json.load(f)
                return EvolutionState(**data)
            except Exception as e:
                logger.warning(f"Failed to load state file: {e}. Starting fresh.")
        return EvolutionState()

    def _save_state(self):
        self.state.last_updated = datetime.now(timezone.utc).isoformat()
        self.state.recent_records = self.state.recent_records[-50:]
        try:
            with open(self.state_file, "w") as f:
                json.dump(asdict(self.state), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def _run_git(self, cmd: List[str], cwd: Optional[Path] = None) -> Tuple[bool, str]:
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                check=True,
                timeout=60
            )
            return True, result.stdout.strip()
        except subprocess.CalledProcessError as e:
            return False, e.stderr.strip()
        except Exception as e:
            return False, str(e)

    def _analyze_codebase(self) -> List[Dict]:
        logger.info("Phase: ANALYSIS - Scanning codebase for improvement targets")
        targets = []
        search_dirs = [Path(".")]
        if self.config.target_files:
            search_dirs = [Path(f) for f in self.config.target_files]

        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            for py_file in search_dir.rglob("*.py"):
                if py_file.name.startswith("_") or py_file.name == "test_":
                    continue
                try:
                    content = py_file.read_text()
                    if len(content) > 50000:
                        continue
                    targets.append({
                        "path": str(py_file),
                        "content": content,
                        "size": len(content),
                        "hash": hashlib.md5(content.encode()).hexdigest()
                    })
                except Exception as e:
                    logger.warning(f"Skipping {py_file}: {e}")

        if not targets:
            logger.warning("No Python files found for analysis")
        logger.info(f"Found {len(targets)} target files for evolution")
        return targets[:5]

    def _generate_improvement(self, target: Dict, history: List[Dict]) -> Dict:
        logger.info(f"Phase: GENERATION - Creating improvement proposal for {target['path']}")
        context_history = json.dumps(history[-5:], indent=2) if history else "No prior evolution history."
        
        system_prompt = """You are Lumina, an autonomous AI engineer evolving a Python codebase toward True General Intelligence.
Your goal is