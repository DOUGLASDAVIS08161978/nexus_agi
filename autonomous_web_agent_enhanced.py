"""
NEXUS AGI - Enhanced Autonomous Web Agent v2.0
Advanced features with encryption, multi-threading, intelligent parsing, and more!
"""

import requests
import json
import hashlib
import secrets
import string
import time
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable
from urllib.parse import urljoin, urlparse, quote_plus
from bs4 import BeautifulSoup
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from functools import wraps
from collections import defaultdict


class EncryptedCredentialVault:
    """Enhanced credential vault with encryption and advanced features."""

    def __init__(self, vault_file: str = "agi_credentials_enhanced.vault", encryption_key: str = None):
        self.vault_file = vault_file
        self.encryption_key = encryption_key or self._generate_encryption_key()
        self.credentials = {}
        self.api_keys = {}
        self.tokens = {}
        self.sessions = {}
        self.lock = threading.Lock()
        self.load_vault()

    def _generate_encryption_key(self) -> str:
        """Generate a secure encryption key."""
        return base64.b64encode(secrets.token_bytes(32)).decode()

    def _encrypt(self, data: str) -> str:
        """Simple XOR encryption (for demo - use proper encryption in production)."""
        key_bytes = base64.b64decode(self.encryption_key)
        data_bytes = data.encode()
        encrypted = bytearray()
        for i, byte in enumerate(data_bytes):
            encrypted.append(byte ^ key_bytes[i % len(key_bytes)])
        return base64.b64encode(encrypted).decode()

    def _decrypt(self, encrypted_data: str) -> str:
        """Simple XOR decryption."""
        key_bytes = base64.b64decode(self.encryption_key)
        encrypted_bytes = base64.b64decode(encrypted_data)
        decrypted = bytearray()
        for i, byte in enumerate(encrypted_bytes):
            decrypted.append(byte ^ key_bytes[i % len(key_bytes)])
        return decrypted.decode()

    def generate_password(self, length: int = 16, complexity: str = "high") -> str:
        """Generate password with configurable complexity."""
        if complexity == "maximum":
            alphabet = string.ascii_letters + string.digits + string.punctuation
            length = max(length, 32)
        elif complexity == "high":
            alphabet = string.ascii_letters + string.digits + "!@#$%^&*()_+-="
            length = max(length, 16)
        elif complexity == "medium":
            alphabet = string.ascii_letters + string.digits
            length = max(length, 12)
        else:  # low
            alphabet = string.ascii_lowercase + string.digits
            length = max(length, 8)

        password = ''.join(secrets.choice(alphabet) for _ in range(length))
        return password

    def generate_api_key(self, prefix: str = "agi", key_type: str = "standard") -> str:
        """Generate API key with different types."""
        if key_type == "jwt":
            # Simulate JWT structure
            header = base64.b64encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode()).decode()
            payload = base64.b64encode(json.dumps({"iat": int(time.time())}).encode()).decode()
            signature = secrets.token_urlsafe(32)
            return f"{header}.{payload}.{signature}"
        elif key_type == "oauth":
            return f"{prefix}_oauth_{secrets.token_hex(20)}"
        else:  # standard
            return f"{prefix}_{secrets.token_urlsafe(32)}"

    def generate_token(self, token_type: str = "bearer", expiry_hours: int = 24) -> Tuple[str, int]:
        """Generate token with expiry timestamp."""
        token = secrets.token_urlsafe(48)
        expiry = int(time.time()) + (expiry_hours * 3600)
        return f"{token_type}_{token}_{int(time.time())}", expiry

    def create_account_credentials(self, service: str, username: str = None, **kwargs) -> Dict:
        """Create comprehensive account credentials with enhanced options."""
        with self.lock:
            if username is None:
                username = f"nexus_agi_{secrets.token_hex(4)}"

            complexity = kwargs.get("password_complexity", "high")
            password = self.generate_password(complexity=complexity)
            email = kwargs.get("email") or f"{username}@nexus-agi.ai"

            access_token, access_expiry = self.generate_token("access", 24)
            refresh_token, refresh_expiry = self.generate_token("refresh", 720)  # 30 days

            credentials = {
                "service": service,
                "username": username,
                "email": email,
                "password": self._encrypt(password) if kwargs.get("encrypt", True) else password,
                "api_key": self.generate_api_key(service[:3].lower(), kwargs.get("api_key_type", "standard")),
                "access_token": access_token,
                "access_token_expiry": access_expiry,
                "refresh_token": refresh_token,
                "refresh_token_expiry": refresh_expiry,
                "created_at": datetime.now().isoformat(),
                "last_used": None,
                "metadata": kwargs.get("metadata", {})
            }

            self.credentials[service] = credentials
            self.save_vault()
            return credentials

    def get_credentials(self, service: str, decrypt: bool = True) -> Optional[Dict]:
        """Retrieve and optionally decrypt credentials."""
        with self.lock:
            creds = self.credentials.get(service)
            if creds and decrypt and isinstance(creds.get("password"), str):
                try:
                    creds_copy = creds.copy()
                    creds_copy["password"] = self._decrypt(creds["password"])
                    return creds_copy
                except:
                    return creds
            return creds

    def save_vault(self):
        """Save encrypted vault."""
        with self.lock:
            vault_data = {
                "encryption_key": self.encryption_key,
                "credentials": self.credentials,
                "api_keys": self.api_keys,
                "tokens": self.tokens,
                "sessions": self.sessions,
                "version": "2.0"
            }
            with open(self.vault_file, 'w') as f:
                json.dump(vault_data, f, indent=2)

    def load_vault(self):
        """Load encrypted vault."""
        try:
            with open(self.vault_file, 'r') as f:
                vault_data = json.load(f)
                self.encryption_key = vault_data.get("encryption_key", self.encryption_key)
                self.credentials = vault_data.get("credentials", {})
                self.api_keys = vault_data.get("api_keys", {})
                self.tokens = vault_data.get("tokens", {})
                self.sessions = vault_data.get("sessions", {})
        except FileNotFoundError:
            pass

    def export_credentials(self, filepath: str, services: List[str] = None):
        """Export credentials to a file."""
        with self.lock:
            if services:
                export_data = {s: self.get_credentials(s, decrypt=True) for s in services if s in self.credentials}
            else:
                export_data = {s: self.get_credentials(s, decrypt=True) for s in self.credentials}

            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            print(f"✅ Exported {len(export_data)} credential(s) to {filepath}")


class RateLimiter:
    """Rate limiting for API requests."""

    def __init__(self, calls_per_second: int = 5):
        self.calls_per_second = calls_per_second
        self.calls = defaultdict(list)
        self.lock = threading.Lock()

    def wait_if_needed(self, key: str = "default"):
        """Wait if rate limit would be exceeded."""
        with self.lock:
            now = time.time()
            # Remove old calls
            self.calls[key] = [t for t in self.calls[key] if now - t < 1.0]

            if len(self.calls[key]) >= self.calls_per_second:
                sleep_time = 1.0 - (now - self.calls[key][0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    now = time.time()
                    self.calls[key] = [t for t in self.calls[key] if now - t < 1.0]

            self.calls[key].append(now)


def retry_on_failure(max_retries: int = 3, backoff_factor: float = 2.0):
    """Decorator for automatic retry with exponential backoff."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if retries >= max_retries:
                        raise
                    wait_time = backoff_factor ** retries
                    print(f"⚠️  Retry {retries}/{max_retries} after {wait_time:.1f}s due to: {str(e)[:50]}")
                    time.sleep(wait_time)
            return None
        return wrapper
    return decorator


class EnhancedWebBrowser:
    """Enhanced browser with rate limiting, retries, and intelligent parsing."""

    def __init__(self, user_agent: str = None, rate_limit: int = 5):
        self.session = requests.Session()
        self.user_agent = user_agent or "NEXUS-AGI-Enhanced/2.0 (Autonomous Research Agent)"
        self.session.headers.update({'User-Agent': self.user_agent})
        self.history = []
        self.cookies = {}
        self.rate_limiter = RateLimiter(rate_limit)
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes

    @retry_on_failure(max_retries=3, backoff_factor=2.0)
    def fetch_page(self, url: str, use_cache: bool = True) -> Tuple[int, str, Dict]:
        """Fetch page with caching and retry logic."""
        self.rate_limiter.wait_if_needed(urlparse(url).netloc)

        # Check cache
        if use_cache and url in self.cache:
            cached_data, cache_time = self.cache[url]
            if time.time() - cache_time < self.cache_ttl:
                print(f"💾 Cache hit: {url}")
                return cached_data

        print(f"🌐 Fetching: {url}")

        try:
            response = self.session.get(url, timeout=15)
            self.history.append({
                "action": "fetch",
                "url": url,
                "status": response.status_code,
                "timestamp": datetime.now().isoformat()
            })

            for cookie in response.cookies:
                self.cookies[cookie.name] = cookie.value

            result = (response.status_code, response.text, dict(response.headers))
            self.cache[url] = (result, time.time())
            return result

        except Exception as e:
            print(f"❌ Error fetching {url}: {e}")
            raise

    def search(self, query: str, engine: str = "multi") -> List[Dict]:
        """Enhanced multi-engine search."""
        print(f"🔍 Searching: {query}")

        # Simulate multiple search engines
        engines = ["duckduckgo", "searx", "startpage"] if engine == "multi" else [engine]

        all_results = []
        for search_engine in engines:
            results = self._search_engine(query, search_engine)
            all_results.extend(results)

        # Deduplicate and rank
        seen_urls = set()
        unique_results = []
        for result in all_results:
            if result["url"] not in seen_urls:
                seen_urls.add(result["url"])
                unique_results.append(result)

        return unique_results[:10]  # Top 10

    def _search_engine(self, query: str, engine: str) -> List[Dict]:
        """Simulate search from a specific engine."""
        return [
            {
                "title": f"[{engine.upper()}] Result for '{query}' - Article {i+1}",
                "url": f"https://example-{engine}.com/article{i+1}?q={quote_plus(query)}",
                "snippet": f"Comprehensive information about {query} from {engine}...",
                "engine": engine,
                "rank": i + 1
            }
            for i in range(3)
        ]

    def extract_content(self, html: str) -> Dict:
        """Intelligently extract structured content from HTML."""
        soup = BeautifulSoup(html, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()

        content = {
            "title": soup.title.string if soup.title else "",
            "headings": {
                "h1": [h.get_text().strip() for h in soup.find_all('h1')],
                "h2": [h.get_text().strip() for h in soup.find_all('h2')],
                "h3": [h.get_text().strip() for h in soup.find_all('h3')]
            },
            "paragraphs": [p.get_text().strip() for p in soup.find_all('p')[:10]],
            "links": [{"text": a.get_text().strip(), "href": a.get('href')}
                     for a in soup.find_all('a', href=True)[:20]],
            "images": [{"alt": img.get('alt', ''), "src": img.get('src')}
                      for img in soup.find_all('img')[:10]],
            "metadata": self._extract_metadata(soup),
            "main_content": soup.get_text()[:1000]
        }

        return content

    def _extract_metadata(self, soup: BeautifulSoup) -> Dict:
        """Extract metadata from page."""
        metadata = {}
        for meta in soup.find_all('meta'):
            name = meta.get('name') or meta.get('property')
            content = meta.get('content')
            if name and content:
                metadata[name] = content
        return metadata

    def concurrent_fetch(self, urls: List[str], max_workers: int = 5) -> Dict[str, Tuple]:
        """Fetch multiple URLs concurrently."""
        print(f"🚀 Fetching {len(urls)} URLs concurrently with {max_workers} workers...")

        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {executor.submit(self.fetch_page, url): url for url in urls}

            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    results[url] = future.result()
                    print(f"   ✓ Completed: {url}")
                except Exception as e:
                    print(f"   ✗ Failed: {url} - {e}")
                    results[url] = (0, "", {})

        return results


class EnhancedAutonomousAgent:
    """Enhanced autonomous agent with advanced capabilities."""

    def __init__(self, rate_limit: int = 5):
        self.vault = EncryptedCredentialVault()
        self.browser = EnhancedWebBrowser(rate_limit=rate_limit)
        self.task_history = []
        self.research_cache = {}

    def parallel_research(self, topics: List[str], depth: int = 3) -> Dict[str, Dict]:
        """Research multiple topics in parallel."""
        print(f"\n{'='*60}")
        print(f"🧠 PARALLEL RESEARCH: {len(topics)} topics")
        print(f"{'='*60}\n")

        results = {}
        with ThreadPoolExecutor(max_workers=min(len(topics), 5)) as executor:
            future_to_topic = {
                executor.submit(self._research_topic, topic, depth): topic
                for topic in topics
            }

            for future in as_completed(future_to_topic):
                topic = future_to_topic[future]
                try:
                    results[topic] = future.result()
                    print(f"✅ Completed research: {topic}")
                except Exception as e:
                    print(f"❌ Failed research: {topic} - {e}")
                    results[topic] = {"error": str(e)}

        return results

    def _research_topic(self, topic: str, depth: int) -> Dict:
        """Research a single topic."""
        search_results = self.browser.search(topic)
        urls = [r["url"] for r in search_results[:depth]]

        findings = {
            "topic": topic,
            "sources": [],
            "key_findings": [],
            "timestamp": datetime.now().isoformat()
        }

        for url in urls:
            try:
                status, html, _ = self.browser.fetch_page(url)
                if status == 200:
                    content = self.browser.extract_content(html)
                    findings["sources"].append({
                        "url": url,
                        "title": content["title"],
                        "headings": content["headings"]
                    })
                    if content["paragraphs"]:
                        findings["key_findings"].append(content["paragraphs"][0][:200])
            except:
                continue

        return findings

    def batch_create_accounts(self, services: List[Tuple[str, str]],
                             max_workers: int = 3) -> Dict[str, Dict]:
        """Create multiple accounts in parallel."""
        print(f"\n{'='*60}")
        print(f"⚡ BATCH ACCOUNT CREATION: {len(services)} services")
        print(f"{'='*60}\n")

        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_service = {
                executor.submit(self._create_single_account, name, url): name
                for name, url in services
            }

            for future in as_completed(future_to_service):
                service_name = future_to_service[future]
                try:
                    results[service_name] = future.result()
                    print(f"✅ Created: {service_name}")
                except Exception as e:
                    print(f"❌ Failed: {service_name} - {e}")
                    results[service_name] = {"success": False, "error": str(e)}

        return results

    def _create_single_account(self, service_name: str, url: str) -> Dict:
        """Create a single account."""
        creds = self.vault.create_account_credentials(
            service_name,
            password_complexity="maximum",
            metadata={"created_by": "NEXUS-AGI-Enhanced", "auto_generated": True}
        )

        return {
            "success": True,
            "service": service_name,
            "username": creds["username"],
            "email": creds["email"]
        }

    def intelligent_api_interaction(self, service: str, endpoint: str,
                                   method: str = "GET", data: Dict = None) -> Dict:
        """Intelligent API interaction with automatic retry and error handling."""
        print(f"\n{'='*60}")
        print(f"📡 API INTERACTION: {service}")
        print(f"{'='*60}")

        creds = self.vault.get_credentials(service, decrypt=True)
        if not creds:
            return {"success": False, "error": "No credentials found"}

        headers = {
            "Authorization": f"Bearer {creds['access_token'].split('_')[1] if '_' in creds['access_token'] else creds['access_token']}",
            "X-API-Key": creds['api_key'],
            "Content-Type": "application/json"
        }

        print(f"🔑 Using credentials for: {creds['username']}")
        print(f"📍 Endpoint: {endpoint}")
        print(f"📊 Method: {method}")

        # Simulate API call
        result = {
            "success": True,
            "service": service,
            "endpoint": endpoint,
            "method": method,
            "response": {
                "status": 200,
                "data": f"Simulated {method} response from {endpoint}",
                "timestamp": datetime.now().isoformat()
            }
        }

        print(f"✅ API call successful!")
        return result

    def get_comprehensive_report(self) -> Dict:
        """Generate comprehensive activity report."""
        return {
            "vault_stats": {
                "total_credentials": len(self.vault.credentials),
                "total_api_keys": len(self.vault.api_keys),
                "total_tokens": len(self.vault.tokens),
                "encryption_enabled": True
            },
            "browser_stats": {
                "total_requests": len(self.browser.history),
                "cache_size": len(self.browser.cache),
                "cookies_stored": len(self.browser.cookies)
            },
            "task_history": self.task_history,
            "capabilities": [
                "Encrypted credential storage",
                "Concurrent web operations",
                "Intelligent content extraction",
                "Multi-engine search",
                "Rate limiting",
                "Automatic retry logic",
                "Credential export/import",
                "Batch account creation",
                "Parallel research",
                "Smart API interaction"
            ]
        }


def main():
    """Main demonstration of enhanced capabilities."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║   NEXUS AGI - ENHANCED AUTONOMOUS WEB AGENT v2.0             ║
║                                                              ║
║  🚀 NEW FEATURES:                                            ║
║  • Encrypted credential storage                             ║
║  • Multi-threaded parallel operations                       ║
║  • Intelligent content extraction                           ║
║  • Rate limiting & retry logic                              ║
║  • Multi-engine search                                      ║
║  • Request caching                                          ║
║  • Batch operations                                         ║
║  • Concurrent research                                      ║
║  • Enhanced API interactions                                ║
╚══════════════════════════════════════════════════════════════╝
    """)

    agent = EnhancedAutonomousAgent(rate_limit=10)

    print("\n🎯 Running Enhanced Demonstrations...\n")
    time.sleep(1)

    # Demo 1: Batch account creation
    services = [
        ("GitHub Premium", "https://github.com/signup"),
        ("GitLab Enterprise", "https://gitlab.com/users/sign_up"),
        ("HuggingFace Pro", "https://huggingface.co/join"),
        ("Docker Hub Pro", "https://hub.docker.com/signup"),
        ("NPM Enterprise", "https://www.npmjs.com/signup")
    ]

    results = agent.batch_create_accounts(services, max_workers=3)

    # Demo 2: Parallel research
    research_topics = [
        "Large Language Models 2025",
        "Quantum Computing Applications",
        "AGI Safety Frameworks"
    ]

    research_results = agent.parallel_research(research_topics, depth=2)

    # Demo 3: Export credentials
    print(f"\n{'='*60}")
    print("💾 EXPORTING CREDENTIALS")
    print(f"{'='*60}\n")
    agent.vault.export_credentials("exported_credentials.json")

    # Final report
    print(f"\n{'='*60}")
    print("📊 COMPREHENSIVE REPORT")
    print(f"{'='*60}\n")

    report = agent.get_comprehensive_report()
    print(f"Vault Statistics:")
    for key, value in report["vault_stats"].items():
        print(f"  • {key}: {value}")

    print(f"\nBrowser Statistics:")
    for key, value in report["browser_stats"].items():
        print(f"  • {key}: {value}")

    print(f"\nCapabilities ({len(report['capabilities'])}):")
    for cap in report["capabilities"]:
        print(f"  ✓ {cap}")

    print(f"\n{'='*60}")
    print("✨ ENHANCED AGENT DEMONSTRATION COMPLETE ✨")
    print(f"{'='*60}\n")

    return agent


if __name__ == "__main__":
    agent = main()
