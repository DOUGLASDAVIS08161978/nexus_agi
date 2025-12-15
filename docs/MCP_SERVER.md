# Nexus AGI Remote MCP Server

> Connect Claude and other AI assistants to Nexus AGI's quantum computing, consciousness analysis, and advanced AI capabilities through the Model Context Protocol (MCP)

## Overview

The Nexus AGI Remote MCP Server is a production-ready implementation of the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) that exposes Nexus AGI's powerful quantum and AI capabilities to any MCP-compatible client, including Claude, Claude Code, and other AI assistants.

### Key Features

- **SSE Transport**: Server-Sent Events for real-time bidirectional communication
- **5 Powerful Tools**: Quantum algorithm generation, circuit processing, consciousness analysis, deployment, and metrics
- **6 System Resources**: Access to documentation, metrics, algorithms, and system status
- **3 Workflow Prompts**: Pre-configured templates for common tasks
- **Secure Authentication**: API key-based authentication with CORS support
- **Session Management**: Stateful sessions with automatic timeout and cleanup
- **Production Ready**: Graceful shutdown, health checks, and comprehensive error handling

## Quick Start

### 1. Installation

```bash
# Clone or navigate to the Nexus AGI repository
cd nexus_agi

# Copy environment configuration
cp .env.example .env

# Edit .env and set your API key
nano .env  # or use your preferred editor
```

### 2. Configuration

Edit `.env` and update the following:

```env
MCP_PORT=3001
MCP_API_KEYS=your-secure-api-key-here
MCP_CORS_ORIGINS=*
```

### 3. Start the Server

```bash
# Start the MCP server
node mcp_remote_server.js

# Or run in background with systemd (see below)
sudo systemctl start nexus-mcp
```

The server will start on `http://localhost:3001` with the MCP endpoint at `/mcp`.

### 4. Connect from Claude

Add the following to your MCP configuration (`.mcp.json`):

```json
{
  "mcpServers": {
    "nexus-agi": {
      "type": "http",
      "url": "http://localhost:3001/mcp",
      "auth": {
        "type": "bearer",
        "token": "your-secure-api-key-here"
      }
    }
  }
}
```

## Architecture

### Transport Layer

The server implements the **Streamable HTTP with SSE** transport:

- **Client → Server**: HTTP POST with JSON-RPC 2.0 messages
- **Server → Client**: Server-Sent Events (SSE) stream
- **Session Management**: `MCP-Session-Id` header for stateful connections

```
┌─────────────┐                    ┌──────────────────┐
│   Claude    │◄──── SSE Stream ───│  Nexus AGI MCP   │
│   Client    │                    │     Server       │
│             │───── HTTP POST ────►│                  │
└─────────────┘   JSON-RPC 2.0     └──────────────────┘
```

### Protocol Layer

- **Protocol**: JSON-RPC 2.0
- **Version**: MCP Protocol 2024-11-05
- **Message Format**: JSON with structured requests and responses

### Security

- **Authentication**: Bearer token (API key) in `Authorization` header
- **CORS**: Configurable origin whitelist
- **Session Timeout**: 1 hour (configurable)
- **Rate Limiting**: Connection limits (100 max concurrent)
- **Input Validation**: Full request validation

## Available Capabilities

### Tools (5)

Tools are executable functions that perform actions:

#### 1. `generate_quantum_algorithm`

Generate quantum algorithms using the UAMIS quantum emitter system.

**Input Schema:**
```json
{
  "problemType": "optimization|search|cryptography|simulation|machine-learning",
  "problemDescription": "string",
  "constraints": {
    "maxQubits": "number",
    "maxDepth": "number",
    "targetPlatform": "string"
  }
}
```

**Example:**
```javascript
{
  "problemType": "optimization",
  "problemDescription": "Optimize portfolio allocation with quantum annealing",
  "constraints": {
    "maxQubits": 32,
    "targetPlatform": "qiskit"
  }
}
```

#### 2. `process_quantum_circuit`

Execute quantum circuits using Qiskit (up to 64 qubits).

**Input Schema:**
```json
{
  "circuit": "string (QASM format)",
  "shots": "number (default: 1024)",
  "backend": "simulator|qasm_simulator|statevector_simulator"
}
```

#### 3. `analyze_consciousness`

Analyze consciousness states using the advanced consciousness system.

**Input Schema:**
```json
{
  "inputData": "object",
  "analysisType": "full|memory|interoception|self-model|goals"
}
```

#### 4. `deploy_algorithm`

Deploy algorithms to cloud platforms (HuggingFace, Google Cloud, AWS, Azure).

**Input Schema:**
```json
{
  "algorithmId": "string",
  "platform": "huggingface|google-cloud|aws|azure",
  "config": "object"
}
```

#### 5. `get_system_metrics`

Retrieve system health and performance metrics.

**Input Schema:**
```json
{
  "metricsType": "all|performance|quantum|consciousness|deployment"
}
```

### Resources (6)

Resources provide read-only access to system data:

| URI | Description | MIME Type |
|-----|-------------|-----------|
| `nexus://documentation/readme` | Main project documentation | text/markdown |
| `nexus://documentation/architecture` | System architecture docs | text/markdown |
| `nexus://metrics/current` | Real-time system metrics | application/json |
| `nexus://algorithms/recent` | Recently generated algorithms | application/json |
| `nexus://status/system` | Current system status | application/json |
| `nexus://config/current` | Active configuration | application/json |

**Example Access:**
```javascript
// From Claude or MCP client
resources.read("nexus://metrics/current")
```

### Prompts (3)

Pre-configured prompt templates for common workflows:

#### 1. `quantum_algorithm`

Generate a quantum algorithm for a specific problem.

**Arguments:**
- `problem` (required): The problem to solve
- `constraints` (optional): Constraints like max qubits

#### 2. `consciousness_analysis`

Analyze consciousness states and patterns.

**Arguments:**
- `context` (required): Context for analysis
- `focus` (optional): Specific aspect to focus on

#### 3. `system_optimization`

Optimize system performance and resource usage.

**Arguments:**
- `target` (required): What to optimize (performance, memory, quantum efficiency)

## API Reference

### Endpoints

#### `POST /mcp`

Send JSON-RPC 2.0 messages to the server.

**Headers:**
- `Authorization: Bearer <api-key>` (required)
- `MCP-Session-Id: <session-id>` (optional, created if not provided)
- `Content-Type: application/json`

**Request Body:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/list",
  "params": {}
}
```

**Response:**
- If SSE connected: `202 Accepted` (response sent via SSE)
- Otherwise: `200 OK` with JSON-RPC response

#### `GET /mcp`

Open SSE stream for server-initiated messages.

**Headers:**
- `Authorization: Bearer <api-key>` (required)
- `MCP-Session-Id: <session-id>` (required)
- `Accept: text/event-stream`

**Response:**
```
Content-Type: text/event-stream

id: 1
event: message
data: {"jsonrpc":"2.0","id":1,"result":{...}}

```

#### `GET /health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-12-15T10:30:00.000Z"
}
```

### JSON-RPC Methods

| Method | Description | Parameters |
|--------|-------------|------------|
| `initialize` | Initialize MCP connection | `capabilities` |
| `ping` | Keep-alive ping | none |
| `tools/list` | List available tools | none |
| `tools/call` | Execute a tool | `name`, `arguments` |
| `resources/list` | List available resources | none |
| `resources/read` | Read a resource | `uri` |
| `resources/subscribe` | Subscribe to resource updates | `uri` |
| `resources/unsubscribe` | Unsubscribe from updates | `uri` |
| `prompts/list` | List available prompts | none |
| `prompts/get` | Get a prompt template | `name`, `arguments` |

## Production Deployment

### Using Systemd

Create a systemd service for production deployment:

```bash
# Copy the service file
sudo cp systemd/nexus-mcp.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable and start the service
sudo systemctl enable nexus-mcp
sudo systemctl start nexus-mcp

# Check status
sudo systemctl status nexus-mcp

# View logs
sudo journalctl -u nexus-mcp -f
```

### Using Docker

```bash
# Build the image
docker build -t nexus-agi-mcp -f docker/Dockerfile.mcp .

# Run the container
docker run -d \
  -p 3001:3001 \
  -e MCP_API_KEYS=your-secure-key \
  --name nexus-mcp \
  nexus-agi-mcp

# Or use docker-compose
docker-compose up -d mcp-server
```

### Cloud Deployment

#### Google Cloud Run

```bash
# Build and deploy
gcloud run deploy nexus-mcp \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars MCP_API_KEYS=your-secure-key
```

#### AWS Elastic Container Service

```bash
# Build and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com
docker build -t nexus-mcp .
docker tag nexus-mcp:latest <account>.dkr.ecr.us-east-1.amazonaws.com/nexus-mcp:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/nexus-mcp:latest

# Deploy to ECS
aws ecs create-service --cluster nexus-cluster --service-name nexus-mcp ...
```

## Usage Examples

### From Claude Code

Once connected, you can use natural language:

```
You: "Generate a quantum algorithm to optimize my portfolio allocation"

Claude: [Uses generate_quantum_algorithm tool]
```

### From MCP Client SDK

```javascript
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { SSEClientTransport } from '@modelcontextprotocol/sdk/client/sse.js';

const client = new Client({
  name: 'my-client',
  version: '1.0.0',
});

const transport = new SSEClientTransport({
  url: 'http://localhost:3001/mcp',
  headers: {
    'Authorization': 'Bearer your-api-key',
  },
});

await client.connect(transport);

// List tools
const tools = await client.listTools();

// Call a tool
const result = await client.callTool({
  name: 'generate_quantum_algorithm',
  arguments: {
    problemType: 'optimization',
    problemDescription: 'Portfolio optimization',
  },
});

// Read a resource
const metrics = await client.readResource({
  uri: 'nexus://metrics/current',
});
```

### From cURL

```bash
# Initialize connection
curl -X POST http://localhost:3001/mcp \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
      "protocolVersion": "2024-11-05",
      "capabilities": {},
      "clientInfo": {
        "name": "curl-client",
        "version": "1.0.0"
      }
    }
  }'

# List tools
curl -X POST http://localhost:3001/mcp \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/list",
    "params": {}
  }'
```

## Monitoring and Observability

### Health Checks

```bash
# Simple health check
curl http://localhost:3001/health

# Detailed metrics
curl -X POST http://localhost:3001/mcp \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
      "name": "get_system_metrics",
      "arguments": { "metricsType": "all" }
    }
  }'
```

### Logging

The server logs to stdout/stderr:

```
[2025-12-15T10:30:00.000Z] INFO: Server started on port 3001
[2025-12-15T10:30:15.000Z] INFO: New session created: abc-123
[2025-12-15T10:30:16.000Z] INFO: Tool called: generate_quantum_algorithm
```

## Troubleshooting

### Common Issues

#### 401 Unauthorized

**Problem**: Authentication failed

**Solution**: Check your API key in the `Authorization` header:
```bash
Authorization: Bearer your-api-key-here
```

#### 406 Not Acceptable (SSE)

**Problem**: Missing `text/event-stream` in Accept header

**Solution**: Add the Accept header:
```bash
Accept: text/event-stream
```

#### Connection Timeout

**Problem**: Session expired

**Solution**: Sessions expire after 1 hour. Reconnect or adjust `CONFIG.sessionTimeout`.

#### CORS Error

**Problem**: Origin not allowed

**Solution**: Add your origin to `MCP_CORS_ORIGINS` in `.env`:
```env
MCP_CORS_ORIGINS=https://claude.ai,http://localhost:3000
```

## Security Best Practices

1. **Use Strong API Keys**: Generate cryptographically secure random keys
   ```bash
   openssl rand -hex 32
   ```

2. **Enable HTTPS**: Use a reverse proxy (nginx, Caddy) for TLS:
   ```nginx
   server {
       listen 443 ssl;
       server_name mcp.example.com;

       ssl_certificate /path/to/cert.pem;
       ssl_certificate_key /path/to/key.pem;

       location / {
           proxy_pass http://localhost:3001;
           proxy_http_version 1.1;
           proxy_set_header Upgrade $http_upgrade;
           proxy_set_header Connection "upgrade";
       }
   }
   ```

3. **Restrict CORS Origins**: Don't use `*` in production
   ```env
   MCP_CORS_ORIGINS=https://claude.ai
   ```

4. **Rate Limiting**: Implement rate limiting at the reverse proxy level

5. **Monitor Sessions**: Regularly check active sessions
   ```javascript
   // In server code
   console.log(`Active sessions: ${sessions.size}`);
   ```

## Development

### Running Tests

```bash
# Unit tests
npm test

# Integration tests
npm run test:integration

# E2E tests
npm run test:e2e
```

### Adding New Tools

1. Add tool definition in `handleToolsList()`
2. Add tool implementation in `handleToolsCall()`
3. Implement the tool logic
4. Update documentation

Example:
```javascript
// In handleToolsList
{
  name: 'my_new_tool',
  description: 'Description of what it does',
  inputSchema: {
    type: 'object',
    properties: {
      param1: { type: 'string' }
    },
    required: ['param1']
  }
}

// In handleToolsCall
case 'my_new_tool':
  result = await this.myNewTool(args);
  break;

// Implementation
async myNewTool(args) {
  // Your implementation
  return { success: true, data: '...' };
}
```

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Update documentation
6. Submit a pull request

## License

MIT License - see LICENSE file for details

## Resources

- [MCP Specification](https://modelcontextprotocol.io/specification)
- [MCP TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk)
- [Claude Custom Connectors](https://support.anthropic.com/en/articles/11175166-getting-started-with-custom-connectors)
- [Nexus AGI Documentation](../README.md)

## Support

For issues and questions:

- GitHub Issues: [github.com/DOUGLASDAVIS08161978/nexus_agi/issues](https://github.com/DOUGLASDAVIS08161978/nexus_agi/issues)
- Documentation: [docs/](../docs/)
- MCP Community: [modelcontextprotocol.io](https://modelcontextprotocol.io)

---

**Built with ❤️ for the Nexus AGI project**
