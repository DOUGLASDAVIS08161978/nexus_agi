# Dependency Audit Report
**Generated:** 2026-01-22
**Project:** DOUGLASDAVIS08161978/nexus_agi

---

## Executive Summary

This audit analyzed dependencies across 3 Node.js packages and 2 Python requirements files. Key findings:

- **29 security vulnerabilities** in root Node.js package (24 low, 5 moderate)
- **Multiple outdated packages** across all Node.js projects
- **Version inconsistencies** between projects
- **Unpinned Python dependencies** creating reproducibility risks
- **Potential bloat** in Python ML/AI stack

**Priority:** Address moderate severity vulnerabilities and standardize dependency versions.

---

## Node.js Dependencies

### 1. Root Package (`/package.json`)

**Project:** nexus-agi-aria v3.0.0

#### Outdated Packages
| Package | Current | Latest | Gap |
|---------|---------|--------|-----|
| `@nomicfoundation/hardhat-toolbox` | 4.0.0 | 6.1.0 | 2 major versions |
| `hardhat` | 2.19.0 | 3.1.5 | 1 major version |
| `dotenv` | 16.3.1 | 17.2.3 | 1 major version |

#### Security Vulnerabilities
- **Total:** 29 vulnerabilities (24 low, 5 moderate, 0 high, 0 critical)
- **Notable Issues:**
  - **Moderate:** Lodash prototype pollution (GHSA-xxjr-mmjv-4gpg)
  - **Moderate:** Various hardhat ignition vulnerabilities
  - **Low:** Multiple elliptic curve cryptography issues
  - **Low:** Cookie parsing vulnerabilities
  - **Low:** Undici resource exhaustion (GHSA-g9mf-h72j-4rw9)

#### Analysis
Most vulnerabilities are in dev dependencies (hardhat tooling). The hardhat ecosystem has many transitive dependencies with older versions. While these are primarily development-time risks, upgrading would reduce attack surface.

#### Recommendations
1. **HIGH PRIORITY:** Upgrade hardhat-toolbox to 6.1.0 (aligns with hashproof-token)
2. **HIGH PRIORITY:** Upgrade hardhat to latest 2.x or evaluate 3.x migration
3. **MEDIUM:** Update dotenv to 17.2.3 for consistency
4. Run `npm audit fix` to auto-resolve fixable vulnerabilities

---

### 2. ARIA Nexus Network (`/aria-nexus-network/package.json`)

**Project:** aria-nexus-network v5.0.0

#### Outdated Packages
| Package | Current | Latest | Gap |
|---------|---------|--------|-----|
| `chalk` | 4.1.2 | 5.6.2 | 1 major version |
| `dotenv` | 16.3.0 | 17.2.3 | 1 major version |

#### Security Vulnerabilities
- **Total:** 0 vulnerabilities

#### Analysis
Clean, minimal dependency tree. Only 3 production dependencies (axios, dotenv, chalk).

**CAUTION:** Chalk v5+ is ESM-only and would require migrating to ES modules or using dynamic imports.

#### Recommendations
1. **MEDIUM:** Update dotenv to 17.2.3 for consistency across projects
2. **LOW:** Consider chalk upgrade only if already using ESM
3. Keep axios at current version (already on latest)

---

### 3. HashProof Token (`/hashproof-token/package.json`)

**Project:** hashproof-token v1.0.0

#### Outdated Packages
- None (all packages at latest versions)

#### Security Vulnerabilities
- **Total:** 0 vulnerabilities

#### Analysis
Most up-to-date of the three Node.js projects. Uses latest hardhat-toolbox (6.1.0) and dotenv (17.2.3).

#### Recommendations
1. **CONSISTENCY:** This project should be the version baseline for others
2. No immediate updates needed

---

## Python Dependencies

### 1. Root Requirements (`/requirements.txt`)

**Purpose:** NEXUS AGI System - ML/AI/Quantum stack

#### Dependency Analysis
```
Core Scientific: numpy, scipy, pandas
ML/DL: torch, transformers, scikit-learn
Quantum: pennylane, qiskit, qiskit-aer
Probabilistic: pyro-ppl
Graph: networkx, torch-geometric
Symbolic: sympy
Visualization: matplotlib
```

#### Issues Identified

**1. Unpinned Versions (High Risk)**
- All packages use `>=` specifiers instead of pinned versions
- **Risk:** Builds are not reproducible
- **Risk:** Breaking changes could be pulled in automatically
- **Example:** `torch>=2.0.0` could pull torch 3.x with breaking changes

**2. Large Install Footprint**
- PyTorch alone: ~2-3GB
- Transformers: ~500MB + models (can be 100GB+)
- Qiskit ecosystem: ~500MB
- **Total estimated:** 5-10GB for full install

**3. Potential Bloat**
If not all features are used:
- Quantum simulation (pennylane, qiskit) may be optional
- Torch-geometric (graph neural networks) niche use case
- Pyro-ppl (probabilistic programming) specialized

**4. Missing Security Scanning**
- No way to check for CVEs without installing packages
- Recommend using `pip-audit` or `safety` tools

#### Recommendations

**HIGH PRIORITY:**
1. **Pin all versions** to specific releases:
   ```
   numpy==1.26.3  # Instead of numpy>=1.24.0
   torch==2.2.0   # Instead of torch>=2.0.0
   ```

2. **Add lock file** using pip-tools or Poetry:
   ```bash
   pip install pip-tools
   pip-compile requirements.txt
   ```

**MEDIUM PRIORITY:**
3. **Create optional dependency groups:**
   ```
   # requirements-core.txt (essential)
   # requirements-quantum.txt (quantum features)
   # requirements-ml.txt (ML features)
   ```

4. **Install and run pip-audit:**
   ```bash
   pip install pip-audit
   pip-audit -r requirements.txt
   ```

**LOW PRIORITY:**
5. Consider lighter alternatives if full features not needed:
   - `torch` → `torch-cpu` (no CUDA, 1/3 the size)
   - `transformers` → specific model classes only

---

### 2. Nexus Pentest Requirements (`/nexus_pentest/requirements.txt`)

**Purpose:** Penetration testing tools

#### Dependency Analysis
```
Network: requests, urllib3, dnspython, scapy, paramiko
Web: beautifulsoup4, lxml
Crypto: pycryptodome
Security: python-nmap, impacket, pyOpenSSL
UI: colorama, tabulate
```

#### Issues Identified

**1. Same Version Pinning Issue**
- All use `>=` specifiers
- Security tools especially need reproducible builds

**2. Potentially Outdated**
Some packages may have CVEs:
- `urllib3>=1.26.0` - version 1.26.x has known issues
- `requests>=2.28.0` - several minor versions behind
- `paramiko>=2.11.0` - SSH library, security-critical

**3. Security Context**
These are security testing tools that:
- Need to be kept current for latest exploits
- Should have locked versions for audit trails
- May have stricter compliance requirements

#### Recommendations

**HIGH PRIORITY:**
1. **Audit with pip-audit** immediately:
   ```bash
   pip-audit -r nexus_pentest/requirements.txt
   ```

2. **Pin all versions** with current latest:
   ```
   requests==2.31.0
   urllib3==2.2.0
   paramiko==3.4.0
   impacket==0.12.0
   ```

**MEDIUM PRIORITY:**
3. **Update regularly** (monthly) for security patches
4. **Document authorization context** in README (CTF, authorized pentesting)

---

## Cross-Project Issues

### Version Inconsistencies

**dotenv versions:**
- Root: 16.3.1
- aria-nexus-network: 16.3.0
- hashproof-token: 17.2.3 ✓

**Recommendation:** Standardize on `17.2.3`

**hardhat-toolbox versions:**
- Root: 4.0.0
- hashproof-token: 6.1.0 ✓

**Recommendation:** Upgrade root to `6.1.0`

### Missing Development Tools

**Recommended additions:**
1. **Node.js:**
   - Add `.nvmrc` or `engines` field for Node version consistency
   - Consider adding `husky` for pre-commit hooks
   - Add `npm-check-updates` for easier updates

2. **Python:**
   - Add `pip-tools` or `Poetry` for dependency management
   - Add `pip-audit` for security scanning
   - Add `.python-version` for pyenv

---

## Bloat Analysis

### Node.js
**Current state:** Only `aria-nexus-network` has installed dependencies

**Potential bloat:**
- Root project has 597 dev dependencies (hardhat toolchain)
- This is normal for Hardhat but consider:
  - Are all hardhat-toolbox features used?
  - Could use granular imports instead of full toolbox

**Recommendation:** Review if all hardhat-toolbox features are needed

### Python

**High bloat risk:**

1. **PyTorch ecosystem:**
   - Full install: ~2-3GB
   - Consider: CPU-only version if no GPU training
   - Consider: Specific model weights only

2. **Transformers:**
   - Library: ~500MB
   - Models can be 100GB+
   - Consider: Install only needed model architectures

3. **Quantum libraries:**
   - PennyLane + Qiskit: ~500MB+
   - Consider: Optional dependency if feature not always used

**Estimated savings:**
- Using torch-cpu: ~1.5GB saved
- Removing unused quantum libs: ~500MB saved
- Total potential: ~2GB reduction

---

## Action Plan

### Immediate Actions (Do This Week)

1. **Security Fixes:**
   ```bash
   cd /home/user/nexus_agi
   npm update @nomicfoundation/hardhat-toolbox
   npm update hardhat
   npm update dotenv
   npm audit fix
   ```

2. **Python Security Audit:**
   ```bash
   pip install pip-audit
   pip-audit -r requirements.txt
   pip-audit -r nexus_pentest/requirements.txt
   ```

3. **Version Consistency:**
   - Update aria-nexus-network dotenv to 17.2.3
   - Standardize hardhat versions

### Short Term (This Month)

4. **Pin Python Dependencies:**
   - Install pip-tools
   - Generate requirements.lock files
   - Test builds with pinned versions

5. **Security Monitoring:**
   - Set up Dependabot or Renovate bot
   - Schedule monthly dependency reviews

6. **Documentation:**
   - Document which Python features are actually used
   - Create optional dependency groups

### Long Term (This Quarter)

7. **Dependency Optimization:**
   - Evaluate if torch-cpu is sufficient
   - Consider splitting Python requirements
   - Review hardhat-toolbox usage

8. **CI/CD Integration:**
   - Add automated security scanning
   - Add dependency update automation
   - Add version consistency checks

---

## Appendix: Update Commands

### Update Root Node.js Project
```bash
cd /home/user/nexus_agi
npm install @nomicfoundation/hardhat-toolbox@^6.1.0
npm install hardhat@latest
npm install dotenv@^17.2.3
npm audit fix --force  # Use with caution
```

### Update ARIA Nexus Network
```bash
cd /home/user/nexus_agi/aria-nexus-network
npm install dotenv@^17.2.3
# Skip chalk update unless migrating to ESM
```

### Pin Python Dependencies
```bash
# Install pip-tools
pip install pip-tools

# Generate locked requirements
cd /home/user/nexus_agi
pip-compile requirements.txt -o requirements.lock
pip-compile nexus_pentest/requirements.txt -o nexus_pentest/requirements.lock

# Install from lock file
pip install -r requirements.lock
```

### Check for Python CVEs
```bash
pip install pip-audit safety

# Using pip-audit (recommended)
pip-audit -r requirements.txt

# Using safety
safety check -r requirements.txt
```

---

## Summary of Recommendations

| Priority | Action | Impact | Effort |
|----------|--------|--------|--------|
| 🔴 HIGH | Pin Python dependencies | Reproducibility | Medium |
| 🔴 HIGH | Audit Python packages for CVEs | Security | Low |
| 🔴 HIGH | Upgrade hardhat-toolbox & hardhat | Security | Low |
| 🟡 MEDIUM | Standardize dotenv versions | Consistency | Low |
| 🟡 MEDIUM | Run npm audit fix | Security | Low |
| 🟡 MEDIUM | Split Python requirements | Flexibility | Medium |
| 🟢 LOW | Evaluate torch-cpu option | Disk space | Medium |
| 🟢 LOW | Review hardhat-toolbox usage | Bloat | High |

---

**Report End**
