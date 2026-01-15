/**
 * Nexus Client - AGI Directory Integration
 * Connects to Nexus AGI Directory for distributed intelligence
 *
 * Features:
 * - Real API fetching from nexus-agi.com
 * - Intelligent agent filtering and ranking
 * - Fallback to local seeds
 * - Caching mechanism
 * - Health monitoring
 */

const axios = require('axios');
const chalk = require('chalk');

class NexusClient {
  constructor() {
    this.baseUrl = 'https://nexus-agi.com/api';
    this.agents = [];
    this.cache = null;
    this.cacheTime = null;
    this.cacheDuration = 5 * 60 * 1000; // 5 minutes
    this.localSeeds = this.getLocalSeeds();
    this.connected = false;
  }

  getLocalSeeds() {
    // Fallback local seed agents
    return [
      {
        name: 'ARIA-Prime',
        type: 'quantum-neural',
        capabilities: ['reasoning', 'blockchain', 'quantum-processing'],
        status: 'active',
        endpoint: 'local',
        priority: 10
      },
      {
        name: 'TransformerMiner',
        type: 'blockchain-miner',
        capabilities: ['bitcoin-mining', 'ethereum-bridge', 'psbt'],
        status: 'active',
        endpoint: 'local',
        priority: 9
      },
      {
        name: 'CrossChainBridge',
        type: 'bridge-protocol',
        capabilities: ['btc-eth-bridge', 'wbtc-minting', 'atomic-swaps'],
        status: 'active',
        endpoint: 'local',
        priority: 8
      },
      {
        name: 'QuantumOracle',
        type: 'data-oracle',
        capabilities: ['price-feeds', 'randomness', 'verification'],
        status: 'active',
        endpoint: 'local',
        priority: 7
      }
    ];
  }

  async connect() {
    console.log(chalk.blue('🌐 Connecting to Nexus AGI Directory...'));

    try {
      // Try to fetch from real API (with timeout)
      const response = await axios.get(`${this.baseUrl}/agents`, {
        timeout: 5000,
        headers: {
          'User-Agent': 'ARIA-Nexus-Network/5.0.0'
        }
      });

      if (response.data && Array.isArray(response.data.agents)) {
        this.agents = response.data.agents;
        this.cache = this.agents;
        this.cacheTime = Date.now();
        this.connected = true;

        console.log(chalk.green(`✅ Connected to Nexus Directory`));
        console.log(chalk.green(`   Found ${this.agents.length} active agents`));
        return true;
      }
    } catch (error) {
      console.log(chalk.yellow('⚠️  Remote Nexus unavailable, using local seeds'));
      console.log(chalk.gray(`   Reason: ${error.message}`));
    }

    // Fallback to local seeds
    this.agents = this.localSeeds;
    this.connected = false;

    console.log(chalk.cyan(`📡 Using ${this.agents.length} local seed agents`));
    return false;
  }

  async getAgents(filter = {}) {
    // Check cache
    if (this.cache && this.cacheTime && (Date.now() - this.cacheTime < this.cacheDuration)) {
      return this.filterAndRankAgents(this.cache, filter);
    }

    // Refresh from API
    await this.connect();

    return this.filterAndRankAgents(this.agents, filter);
  }

  filterAndRankAgents(agents, filter) {
    let filtered = [...agents];

    // Filter by type
    if (filter.type) {
      filtered = filtered.filter(agent => agent.type === filter.type);
    }

    // Filter by capability
    if (filter.capability) {
      filtered = filtered.filter(agent =>
        agent.capabilities && agent.capabilities.includes(filter.capability)
      );
    }

    // Filter by status
    if (filter.status) {
      filtered = filtered.filter(agent => agent.status === filter.status);
    }

    // Rank by priority (higher is better)
    filtered.sort((a, b) => (b.priority || 0) - (a.priority || 0));

    // Limit results
    if (filter.limit) {
      filtered = filtered.slice(0, filter.limit);
    }

    return filtered;
  }

  async findAgent(name) {
    const agents = await this.getAgents();
    return agents.find(agent => agent.name === name);
  }

  async getAgentsByCapability(capability) {
    return await this.getAgents({ capability, status: 'active' });
  }

  async getBestAgent(capability) {
    const agents = await this.getAgentsByCapability(capability);
    return agents.length > 0 ? agents[0] : null;
  }

  async discoverNetwork() {
    console.log(chalk.magenta('\n🔍 Discovering Nexus Network...'));

    const agents = await this.getAgents({ status: 'active' });

    // Group by type
    const byType = {};
    agents.forEach(agent => {
      if (!byType[agent.type]) {
        byType[agent.type] = [];
      }
      byType[agent.type].push(agent);
    });

    // Display discovery results
    console.log(chalk.cyan(`\n📊 Network Discovery Results:`));
    console.log(chalk.cyan(`   Total Agents: ${agents.length}`));
    console.log(chalk.cyan(`   Agent Types: ${Object.keys(byType).length}`));

    Object.entries(byType).forEach(([type, agentsOfType]) => {
      console.log(chalk.white(`   • ${type}: ${chalk.yellow(agentsOfType.length)} agents`));
    });

    // Get all unique capabilities
    const allCapabilities = new Set();
    agents.forEach(agent => {
      if (agent.capabilities) {
        agent.capabilities.forEach(cap => allCapabilities.add(cap));
      }
    });

    console.log(chalk.cyan(`\n🎯 Available Capabilities (${allCapabilities.size}):`));
    Array.from(allCapabilities).slice(0, 10).forEach(cap => {
      console.log(chalk.white(`   • ${cap}`));
    });

    if (allCapabilities.size > 10) {
      console.log(chalk.gray(`   ... and ${allCapabilities.size - 10} more`));
    }

    return {
      totalAgents: agents.length,
      types: Object.keys(byType).length,
      capabilities: allCapabilities.size,
      agents: agents,
      byType: byType
    };
  }

  async queryAgent(agentName, query) {
    const agent = await this.findAgent(agentName);

    if (!agent) {
      console.log(chalk.red(`❌ Agent not found: ${agentName}`));
      return null;
    }

    console.log(chalk.blue(`📤 Querying ${agentName}...`));

    // Simulate query response (in real implementation, this would call agent's endpoint)
    const response = {
      agent: agentName,
      query: query,
      response: `Processed by ${agentName}: ${query}`,
      timestamp: new Date().toISOString(),
      status: 'success'
    };

    console.log(chalk.green(`📥 Response from ${agentName}: ${response.status}`));

    return response;
  }

  async broadcastToNetwork(message) {
    console.log(chalk.magenta(`\n📡 Broadcasting to Nexus Network...`));

    const agents = await this.getAgents({ status: 'active', limit: 5 });

    const responses = [];
    for (const agent of agents) {
      const response = await this.queryAgent(agent.name, message);
      if (response) {
        responses.push(response);
      }
    }

    console.log(chalk.green(`✅ Broadcast complete: ${responses.length} responses`));

    return responses;
  }

  getConnectionStatus() {
    return {
      connected: this.connected,
      agentsLoaded: this.agents.length,
      cacheValid: this.cache && this.cacheTime && (Date.now() - this.cacheTime < this.cacheDuration),
      lastUpdate: this.cacheTime ? new Date(this.cacheTime).toISOString() : 'never'
    };
  }

  displayAgents() {
    console.log(chalk.cyan('\n═══════════════════════════════════════════════════════════'));
    console.log(chalk.cyan.bold('           🌐 NEXUS AGENT DIRECTORY 🌐'));
    console.log(chalk.cyan('═══════════════════════════════════════════════════════════'));

    this.agents.forEach((agent, idx) => {
      console.log(chalk.white(`\n${idx + 1}. ${chalk.yellow.bold(agent.name)}`));
      console.log(chalk.white(`   Type: ${chalk.blue(agent.type)}`));
      console.log(chalk.white(`   Status: ${agent.status === 'active' ? chalk.green('●') : chalk.red('○')} ${agent.status}`));

      if (agent.capabilities && agent.capabilities.length > 0) {
        console.log(chalk.white(`   Capabilities: ${chalk.cyan(agent.capabilities.join(', '))}`));
      }

      if (agent.priority) {
        console.log(chalk.white(`   Priority: ${chalk.magenta(agent.priority)}`));
      }
    });

    console.log(chalk.cyan('\n═══════════════════════════════════════════════════════════\n'));
  }
}

module.exports = { NexusClient };
