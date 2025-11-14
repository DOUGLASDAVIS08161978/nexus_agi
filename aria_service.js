#!/usr/bin/env node
/**
 * ARIA Service - Continuous Execution Loop
 * Runs the ARIA system as a service, processing queries continuously.
 */

const { ARIASystem } = require('./aria.js');
const fs = require('fs');
const path = require('path');

// Configuration
const CONFIG = {
    loopInterval: 300000, // 5 minutes in milliseconds
    logFile: 'aria_service.log',
    queriesPerCycle: 3
};

// Global flag for graceful shutdown
let running = true;

// Logging helper
function log(level, message) {
    const timestamp = new Date().toISOString();
    const logMessage = `${timestamp} - ${level.toUpperCase()} - ${message}\n`;
    
    // Write to console
    console.log(logMessage.trim());
    
    // Write to file
    try {
        fs.appendFileSync(CONFIG.logFile, logMessage);
    } catch (err) {
        console.error(`Failed to write to log file: ${err.message}`);
    }
}

// Signal handlers for graceful shutdown
process.on('SIGINT', () => {
    log('info', 'Received SIGINT. Initiating graceful shutdown...');
    running = false;
});

process.on('SIGTERM', () => {
    log('info', 'Received SIGTERM. Initiating graceful shutdown...');
    running = false;
});

class ARIAService {
    constructor(loopInterval = CONFIG.loopInterval) {
        this.loopInterval = loopInterval;
        this.cycleCount = 0;
        this.startTime = null;
        this.aria = null;
        this.totalQueriesProcessed = 0;
    }

    initialize() {
        log('info', '='.repeat(80));
        log('info', 'ARIA SERVICE - INITIALIZING');
        log('info', '='.repeat(80));
        
        try {
            this.aria = new ARIASystem();
            this.startTime = Date.now();
            log('info', 'ARIA system initialized successfully');
            return true;
        } catch (error) {
            log('error', `Failed to initialize ARIA system: ${error.message}`);
            log('error', error.stack);
            return false;
        }
    }

    generateQuery() {
        // Sample queries for continuous processing
        const queries = [
            "What is the optimal path to sustainable technological advancement?",
            "How can we balance individual freedom with collective wellbeing?",
            "What are the implications of artificial consciousness?",
            "How should humanity approach existential risks?",
            "What is the nature of ethical decision-making in complex systems?",
            "How can we ensure equitable distribution of advanced technologies?",
            "What role should AI play in human decision-making?",
            "How do we maintain human agency in an AI-augmented world?",
            "What are the long-term consequences of climate change mitigation strategies?",
            "How can we preserve human values through rapid technological change?"
        ];
        
        const index = this.totalQueriesProcessed % queries.length;
        return queries[index];
    }

    async processQuery(query) {
        try {
            log('info', `Processing query: "${query}"`);
            
            const response = this.aria.processQuery(query);
            
            // Log key metrics
            log('info', `  → Quantum Confidence: ${(response.quantumAnalysis.quantumConfidence * 100).toFixed(1)}%`);
            log('info', `  → Outcome Quality: ${(response.multiverseInsights.outcomeQuality * 100).toFixed(1)}%`);
            log('info', `  → Timeline Integrity: ${(response.temporalStatus.timelineIntegrity * 100).toFixed(1)}%`);
            log('info', `  → Awareness Level: ${response.consciousExperience.awarenessLevel}`);
            log('info', `  → Synthesis: ${response.synthesis.substring(0, 100)}...`);
            
            this.totalQueriesProcessed++;
            return true;
        } catch (error) {
            log('error', `Error processing query: ${error.message}`);
            log('error', error.stack);
            return false;
        }
    }

    async processCycle() {
        this.cycleCount++;
        log('info', '\n' + '='.repeat(80));
        log('info', `PROCESSING CYCLE #${this.cycleCount}`);
        log('info', '='.repeat(80) + '\n');
        
        try {
            // Process multiple queries per cycle
            for (let i = 0; i < CONFIG.queriesPerCycle; i++) {
                if (!running) break;
                
                const query = this.generateQuery();
                await this.processQuery(query);
                
                // Brief pause between queries
                if (i < CONFIG.queriesPerCycle - 1) {
                    await this.sleep(2000);
                }
            }
            
            return true;
        } catch (error) {
            log('error', `Error in cycle ${this.cycleCount}: ${error.message}`);
            log('error', error.stack);
            return false;
        }
    }

    getStatus() {
        if (this.startTime) {
            const uptime = (Date.now() - this.startTime) / 1000;
            return {
                status: 'running',
                cycles_completed: this.cycleCount,
                queries_processed: this.totalQueriesProcessed,
                uptime_seconds: uptime,
                start_time: new Date(this.startTime).toISOString()
            };
        }
        return { status: 'not_started' };
    }

    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    async run() {
        if (!this.initialize()) {
            log('error', 'Service initialization failed. Exiting.');
            return 1;
        }
        
        log('info', `Service started. Loop interval: ${this.loopInterval / 1000} seconds`);
        log('info', 'Press Ctrl+C to stop gracefully\n');
        
        while (running) {
            try {
                // Process a cycle
                const success = await this.processCycle();
                
                if (!success) {
                    log('warning', 'Cycle processing had errors, but continuing...');
                }
                
                // Log status
                const status = this.getStatus();
                log('info', `Status: ${status.cycles_completed} cycles, ` +
                           `${status.queries_processed} queries, ` +
                           `uptime: ${Math.floor(status.uptime_seconds)}s`);
                
                // Wait before next cycle
                if (running) {
                    log('info', `Waiting ${this.loopInterval / 1000} seconds until next cycle...\n`);
                    
                    // Sleep in short intervals to check for shutdown signal
                    const sleepStep = 1000; // 1 second
                    let sleptTime = 0;
                    
                    while (running && sleptTime < this.loopInterval) {
                        await this.sleep(sleepStep);
                        sleptTime += sleepStep;
                    }
                }
                
            } catch (error) {
                log('error', `Unexpected error in service loop: ${error.message}`);
                log('error', error.stack);
                
                if (running) {
                    log('info', 'Recovering... waiting 30 seconds before retry');
                    await this.sleep(30000);
                }
            }
        }
        
        // Shutdown
        log('info', '\n' + '='.repeat(80));
        log('info', 'ARIA SERVICE - SHUTTING DOWN');
        log('info', '='.repeat(80));
        
        const status = this.getStatus();
        log('info', `Total cycles completed: ${status.cycles_completed}`);
        log('info', `Total queries processed: ${status.queries_processed}`);
        log('info', `Total uptime: ${Math.floor(status.uptime_seconds)} seconds`);
        log('info', 'Service stopped gracefully');
        
        return 0;
    }
}

// Main entry point
async function main() {
    // Parse command line arguments
    const args = process.argv.slice(2);
    let interval = CONFIG.loopInterval;
    
    for (let i = 0; i < args.length; i++) {
        if (args[i] === '--interval' && args[i + 1]) {
            interval = parseInt(args[i + 1]) * 1000; // Convert seconds to milliseconds
            i++;
        } else if (args[i] === '--help') {
            console.log('ARIA Continuous Service');
            console.log('Usage: node aria_service.js [options]');
            console.log('');
            console.log('Options:');
            console.log('  --interval <seconds>  Seconds between processing cycles (default: 300)');
            console.log('  --help               Show this help message');
            process.exit(0);
        }
    }
    
    // Create and run service
    const service = new ARIAService(interval);
    const exitCode = await service.run();
    
    process.exit(exitCode);
}

// Run the service
if (require.main === module) {
    main().catch(error => {
        log('error', `Fatal error: ${error.message}`);
        log('error', error.stack);
        process.exit(1);
    });
}

module.exports = { ARIAService };
