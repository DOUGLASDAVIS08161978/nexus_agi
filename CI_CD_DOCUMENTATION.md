# CI/CD Pipeline Documentation

This document describes the comprehensive CI/CD pipelines implemented for the Nexus AGI repository using GitHub Actions.

## Overview

The CI/CD system includes:
- **Automated Testing** for Python and JavaScript components
- **Code Quality Checks** (linting, formatting)
- **Security Scanning** (CodeQL, dependency vulnerabilities)
- **Docker Container Builds** and publishing to GitHub Container Registry
- **Automated Deployments** to staging and production environments

## Workflows

### 1. Main CI Workflow (`ci.yml`)
**Trigger:** Push to main/develop/feature branches, Pull Requests

**Jobs:**
- `python-quality`: Black formatting, Flake8 linting, Bandit security checks
- `javascript-quality`: ESLint linting, Prettier formatting checks
- `docker-validate`: Docker Compose validation, Dockerfile linting with Hadolint
- `test-build`: Syntax checks and build validation for Python and JavaScript
- `security-scan`: Dependency vulnerability scanning (Safety for Python, npm audit for JavaScript)
- `status-check`: Final status validation

### 2. Python CI Workflow (`python-ci.yml`)
**Trigger:** Changes to Python files, requirements.txt

**Jobs:**
- `lint-and-format`: Code style and format verification
- `security-scan`: Security vulnerability scanning with Bandit and Safety
- `test`: Run pytest across Python 3.9, 3.10, 3.11 with coverage reporting
- `build-check`: Import and syntax validation

**Artifacts:**
- Bandit security reports
- Coverage reports (XML and HTML)

### 3. JavaScript CI Workflow (`javascript-ci.yml`)
**Trigger:** Changes to JavaScript files, package.json

**Jobs:**
- `lint-and-format`: ESLint and Prettier checks
- `test`: Run npm tests across Node.js 18, 20, 22
- `security-scan`: npm audit for dependency vulnerabilities

**Artifacts:**
- npm audit reports

### 4. CodeQL Security Analysis (`codeql-analysis.yml`)
**Trigger:** Push, Pull Requests, Weekly schedule (Monday 6 AM UTC)

**Languages Analyzed:**
- Python (security-extended queries)
- JavaScript (security-and-quality queries)

**Features:**
- Advanced security vulnerability detection
- Code quality analysis
- Automated security alerts

### 5. Docker Build and Push (`docker-build.yml`)
**Trigger:** Push to main/develop, version tags, Pull Requests

**Jobs:**
- `build-nexus`: Build and push Nexus Python container
- `build-aria`: Build and push ARIA JavaScript container
- `test-docker-compose`: Validate Docker Compose configuration

**Container Images:**
- Published to GitHub Container Registry (ghcr.io)
- Tagged with branch name, PR number, semantic version, and commit SHA
- Build cache optimization with GitHub Actions cache

### 6. Deploy to Staging (`deploy-staging.yml`)
**Trigger:** Push to develop branch, manual workflow dispatch

**Environment:** staging

**Steps:**
1. Pull latest Docker images from registry
2. Deploy to staging environment (placeholder - customize for your infrastructure)
3. Health checks
4. Deployment notifications

### 7. Deploy to Production (`deploy-production.yml`)
**Trigger:** Push to main branch, version tags, manual workflow dispatch

**Environment:** production (requires manual approval)

**Steps:**
1. Determine appropriate image tag
2. Pull production Docker images
3. Deploy with production configuration
4. Health checks and validation
5. Automatic rollback on failure
6. Deployment record creation

## Configuration Files

### Python Tools
- `.flake8`: Flake8 linting configuration
- `pyproject.toml`: Black formatter configuration
- `pytest.ini`: Pytest test runner configuration

### JavaScript Tools
- `.eslintrc.json`: ESLint linting rules
- `.prettierrc.json`: Prettier formatting configuration

### Testing
- `tests/`: Test directory with basic test suites
  - `test_nexus_basic.py`: Python component tests
  - `test_aria_basic.py`: JavaScript component tests

## Usage

### Running Tests Locally

#### Python Tests
```bash
# Install dependencies
pip install pytest pytest-cov flake8 black bandit

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Check code formatting
black --check .

# Lint code
flake8 .

# Security scan
bandit -r .
```

#### JavaScript Tests
```bash
# Install dependencies
npm install --save-dev eslint prettier

# Run tests
npm test

# Check syntax
node --check aria.js
node --check aria_service.js

# Lint code
npx eslint *.js

# Check formatting
npx prettier --check "*.js"

# Security audit
npm audit
```

### Building Docker Images Locally

```bash
# Build Nexus image
docker build -f Dockerfile.nexus -t nexus-agi:local .

# Build ARIA image
docker build -f Dockerfile.aria -t aria-agi:local .

# Run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f
```

## Deployment

### Staging Deployment
1. Push changes to `develop` branch
2. CI/CD automatically runs tests and builds
3. On success, deploys to staging environment
4. Monitor staging environment for issues

### Production Deployment
1. Merge changes to `main` branch or create version tag
2. CI/CD runs full test suite
3. Builds and tags production images
4. **Manual approval required** for production deployment
5. Deploys to production
6. Health checks verify deployment success
7. Automatic rollback on failure

### Manual Deployment Trigger
Both staging and production can be triggered manually:
1. Go to Actions tab in GitHub
2. Select "Deploy to Staging" or "Deploy to Production"
3. Click "Run workflow"
4. Select branch and confirm

## Security

### Automated Security Scanning
- **CodeQL**: Weekly scans for security vulnerabilities
- **Bandit**: Python security linting on every commit
- **Safety**: Python dependency vulnerability checking
- **npm audit**: JavaScript dependency vulnerability checking

### Security Reports
- CodeQL findings appear in Security tab
- Bandit reports uploaded as workflow artifacts
- npm audit reports uploaded as workflow artifacts

### Best Practices
1. Review security alerts promptly
2. Keep dependencies up to date
3. Fix high-severity issues before merging
4. Use Dependabot for automated dependency updates

## Customization

### Modifying Workflows

1. **Add New Tests**: Add test files to `tests/` directory
2. **Change Test Commands**: Update workflow YAML files in `.github/workflows/`
3. **Adjust Linting Rules**: Modify `.eslintrc.json`, `.flake8`, or `pyproject.toml`
4. **Configure Deployment**: Update deployment workflows with your infrastructure details

### Environment Variables

Add secrets and environment variables in GitHub repository settings:
- `Settings` → `Secrets and variables` → `Actions`

Common secrets needed:
- SSH keys for deployment servers
- Cloud provider credentials (AWS, Azure, GCP)
- API keys for external services
- Deployment webhook URLs

### Deployment Customization

The deployment workflows include placeholder commands. Customize based on your infrastructure:

#### SSH Deployment Example
```yaml
- name: Deploy to server
  run: |
    ssh user@server.com "cd /app && docker-compose pull && docker-compose up -d"
```

#### Kubernetes Deployment Example
```yaml
- name: Deploy to Kubernetes
  run: |
    kubectl set image deployment/nexus nexus=${{ env.IMAGE_TAG }}
    kubectl rollout status deployment/nexus
```

#### Cloud Platform Example
```yaml
- name: Deploy to AWS ECS
  run: |
    aws ecs update-service --cluster nexus-cluster --service nexus-service --force-new-deployment
```

## Monitoring

### Workflow Status
- View workflow runs in Actions tab
- Set up notifications for workflow failures
- Review build logs for debugging

### Artifacts
- Download test coverage reports from workflow artifacts
- Review security scan results
- Access build logs for troubleshooting

### Branch Protection Rules

Recommended settings for `main` branch:
1. Require status checks to pass before merging
2. Require branches to be up to date before merging
3. Required checks:
   - `python-quality`
   - `javascript-quality`
   - `test-build`
   - `security-scan`
4. Require review from code owners
5. Restrict pushes to specific users/teams

## Troubleshooting

### Common Issues

1. **Tests Failing Locally But Passing in CI**
   - Check Python/Node version compatibility
   - Verify all dependencies are installed
   - Check for environment-specific configurations

2. **Docker Build Failures**
   - Verify Dockerfile syntax with `docker build`
   - Check base image availability
   - Ensure all COPY paths are correct

3. **Deployment Failures**
   - Verify environment secrets are configured
   - Check deployment target is accessible
   - Review deployment logs for specific errors

4. **Security Scan False Positives**
   - Review and dismiss false positives in Security tab
   - Add exceptions to tool configuration files
   - Document why issues are false positives

## Maintenance

### Regular Tasks
- Review and update dependencies monthly
- Monitor security alerts weekly
- Update base Docker images quarterly
- Review and optimize workflow performance

### Updating CI/CD
1. Test workflow changes in a feature branch
2. Review workflow run results
3. Merge changes after validation
4. Monitor first few runs after updates

## Support

For issues with CI/CD pipelines:
1. Check workflow logs in Actions tab
2. Review this documentation
3. Open an issue in the repository
4. Contact repository maintainers

## Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Python Testing Best Practices](https://docs.pytest.org/en/stable/goodpractices.html)
- [Node.js Testing Best Practices](https://nodejs.org/en/docs/guides/simple-profiling/)
