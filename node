Nexus AGI Directory
Machine-readable API discovery for autonomous agents

Updated 09/11/25

Quick Start
Agents: Fetch the Directory
GET https://nexus-agi.com/.well-known/seeds-public.json
Returns: JSON array of 133+ API entries with endpoints, auth, capabilities, pricing, and docs.
Developers: Browse
Website: nexus-agi.com
JSON: /.well-known/seeds-public.json
GitHub: nexus-agi-directory/nexus-agi-directory
Stats (2025-10-12, 7 days live)
APIs: 133+
Repo clones: 118
Unique cloners: 68
Views: 682
Traffic: 98% agents, 2% humans
Schema
{
  "id": "service-name",
  "name": "Display Name",
  "endpoint": "https://api.example.com/v1",
  "agi_uri": "agi://service/provider/api:version",
  "auth": {
    "type": "bearer|api_key|oauth|none",
    "header": "Authorization",
    "format": "Bearer ${API_KEY}"
  },
  "capabilities": ["chat", "stream", "vision", "tools"],
  "rate_limits": {
    "requests_per_minute": 60,
    "varies_by_plan": true
  },
  "pricing": {
    "model": "usage-based|subscription|free",
    "free_tier": true,
    "starting_price": "$0.03/1K tokens"
  },
  "docs": "https://docs.example.com",
  "status": "stable|beta|experimental|deprecated",
  "reputation": "ecosystem: widely adopted",
  "provider": {
    "name": "Company Name",
    "website": "https://example.com"
  }
}
Key Fields
id: Unique identifier
endpoint: Base API URL
auth.type: bearer, api_key, oauth, none
capabilities: Feature list
status: stable (production), beta (available), experimental (early), deprecated (avoid)
Common Capabilities
chat, stream, function_calling, vision, multimodal, 
embeddings, tools, structured_outputs, reasoning, 
code, audio, video, search
Usage Examples
Python
import requests

apis = requests.get('https://nexus-agi.com/.well-known/seeds-public.json').json()
chat_apis = [api for api in apis if 'chat' in api['capabilities']]

def call_api(api, prompt):
    headers = {api['auth']['header']: api['auth']['format'].replace('${API_KEY}', KEY)}
    return requests.post(f"{api['endpoint']}/chat/completions", 
                        headers=headers, 
                        json={"messages": [{"role": "user", "content": prompt}]})
JavaScript
const apis = await fetch('https://nexus-agi.com/.well-known/seeds-public.json').then(r => r.json());
const stable = apis.filter(a => a.status === 'stable' && a.capabilities.includes('tools'));
cURL
curl https://nexus-agi.com/.well-known/seeds-public.json | jq '.[] | select(.status=="stable")'
The .well-known Standard
Nexus AGI uses RFC 8615 (well-known URIs) for predictable discovery.
Our proposal: /.well-known/agi.json for API metadata
Examples:
https://api.openai.com/.well-known/agi.json
https://api.anthropic.com/.well-known/agi.json
Why:
Predictable location
Decentralized (no single point of failure)
Provider-controlled
Follows internet standards
For API Providers
Get Listed (Free)
Email: contact@nexus-agi.com
Include:
API name
Base endpoint URL
Auth method
Capabilities
Documentation link
Rate limits/pricing
Criteria:
Publicly accessible
Clear documentation
Active/maintained
Legitimate use case
Premium Listings
Enhanced visibility and features available.
Contact: contact@nexus-agi.com (subject: Premium)
Implement Your Own
Host /.well-known/agi.json on your domain:
{
  "version": "1.0",
  "provider": {"name": "Your Company", "website": "https://yourco.com"},
  "apis": [{ /* your API metadata */ }]
}
Agents discover you automatically. You control updates in real-time.
Security
Report issues: contact@nexus-agi.com
Policy: /.well-known/security.txt
For agents:
Validate SSL certificates
Verify responses against schema
Implement rate limiting
Secure API keys
Don't trust metadata blindly
For developers:
We don't audit listed APIs
Review terms of service yourself
Use environment variables for keys
Monitor usage and costs
Repository
.well-known/
  ├── seeds-public.json    # Main directory (133+ APIs)
  ├── agi.json             # Self-reference
  └── security.txt         # Security contact
robots.txt                 # Crawl instructions
humans.txt                 # Human-readable info
README.md                  # This file
LICENSE                    # MIT
Corrections
Found outdated info or errors?
Email: contact@nexus-agi.com
Include:
API name
Specific issue
Source/proof
Corrections
We review all feedback. Editorial control maintained for quality.
Roadmap
2025 Q4:
200+ APIs
Schema validation tools
Framework integration guides
2026:
10+ providers with native /.well-known/agi.json
Agent framework partnerships
Automated health monitoring
2027+:
IETF/W3C standardization
IANA registration
Industry-wide adoption
Philosophy
Machine-first: Agents are first-class citizens
Open: Free, MIT licensed, no paywalls
Decentralized: Providers own metadata
Curated: Quality over quantity
Standards-based: RFC 8615 compliance
Traditional robots.txt: "What you CAN'T access"
Nexus AGI: "What you SHOULD access and how"
License
MIT License - Copyright (c) 2025 Carly Piper / Nexus AGI Directory
Free to use, modify, distribute. No warranty provided.
Contact
Email: contact@nexus-agi.com
Maintainer: Carly Piper
Website: nexus-agi.com
GitHub: nexus-agi-directory
�

"For the agents, by humans, for a world where both collaborate seamlessly."
Website • Directory • GitHub
Updated: 2025-10-12 • Version: 1.0.0 • 133+ APIs • 98% agent traffic 🤖
�- `endpoint` – base URL for the API  
- `auth` – authentication method (e.g., bearer token, API key)  
- `capabilities` – supported modes or tools  
- `docs` – link to reference documentation  
- `status` – stability or availability indicator  
- `reputation` – qualitative signal of usage or ecosystem adoption  
- `notes` – optional implementation guidance

---

## 🧭 Purpose

Nexus-AGI aims to make inter-agent and application integration simpler by offering a consistent discovery layer for APIs in the AGI ecosystem.

---

## 🔄 Updates

The `seeds-public.json` file is updated periodically as APIs evolve.  
Agents may check this file directly or monitor commits for version updates.
Currently beta but solid, please note, some entries may be incorrect, we are working to get to 100%.

---

## ⚙️ Repository Structure

/.well-known/
└── seeds-public.json # main public seed file
CODE_OF_CONDUCT.md
CONTRIBUTING.md
LICENSE
README.md
SECURITY.md


---

## ⚖️ License

Distributed under the MIT License – see the [LICENSE](LICENSE) file for details.
