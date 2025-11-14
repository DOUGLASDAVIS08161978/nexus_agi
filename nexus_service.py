sudo su 
#!/usr/bin/env python3
"""
Nexus AGI Service - Continuous Execution Loop
Runs the Nexus AGI system as a service, processing problems continuously.
"""

import sys
import time
import signal
import logging
from datetime import datetime
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import after path is set
from nexus_agi import MetaAlgorithm_NexusCore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nexus_service.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('NexusService')

# Global flag for graceful shutdown
running = True

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global running
    logger.info(f"Received signal {signum}. Initiating graceful shutdown...")
    running = False

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

class NexusService:
    """Service wrapper for continuous Nexus AGI execution"""
    
    def __init__(self, loop_interval=300):
        """
        Initialize the service
        
        Args:
            loop_interval: Seconds to wait between problem solving cycles (default: 5 minutes)
        """
        self.loop_interval = loop_interval
        self.cycle_count = 0
        self.start_time = None
        self.core = None
        
    def initialize(self):
        """Initialize the Nexus Core system"""
        logger.info("=" * 80)
        logger.info("NEXUS AGI SERVICE - INITIALIZING")
        logger.info("=" * 80)
        
        try:
            self.core = MetaAlgorithm_NexusCore()
            self.start_time = datetime.now()
            logger.info("Nexus Core initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Nexus Core: {e}", exc_info=True)
            return False
    
    def process_problem(self):
        """Process a problem using the Nexus system"""
        self.cycle_count += 1
        logger.info(f"\n{'=' * 80}")
        logger.info(f"PROCESSING CYCLE #{self.cycle_count}")
        logger.info(f"{'=' * 80}\n")
        
        try:
            # Define problem for this cycle
            # In a real deployment, this would come from a queue or API
            problem = {
                "title": f"Complex Problem Analysis - Cycle {self.cycle_count}",
                "type": "complex_adaptive_system",
                "domain_knowledge": {
                    "analysis": f"Analyzing complex systems and patterns at {datetime.now().isoformat()}",
                    "optimization": "Seeking optimal solutions through meta-algorithmic composition",
                    "integration": "Integrating multiple knowledge domains for comprehensive solutions"
                },
                "stakeholders": ["system_users", "affected_parties", "future_generations"]
            }
            
            # Define constraints
            constraints = {
                "complexity": 0.7,
                "time_efficiency": 0.8,
                "resource_usage": 0.6
            }
            
            logger.info(f"Problem: {problem['title']}")
            
            # Generate solution
            solution = self.core.solve_complex_problem(problem, constraints)
            
            # Log results
            logger.info("\n" + "=" * 80)
            logger.info("SOLUTION SUMMARY")
            logger.info("=" * 80)
            logger.info(f"Problem: {solution['problem']}")
            logger.info(f"Approach: {solution['approach']}")
            logger.info(f"Subproblems: {len(solution['subproblems'])}")
            logger.info(f"Ethics Score: {solution['ethics_assessment']['ethics_score']:.4f}")
            logger.info(f"Effectiveness: {solution['estimated_performance']['effectiveness']:.4f}")
            logger.info("=" * 80 + "\n")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing problem in cycle {self.cycle_count}: {e}", exc_info=True)
            return False
    
    def get_status(self):
        """Get current service status"""
        if self.start_time:
            uptime = datetime.now() - self.start_time
            return {
                "status": "running",
                "cycles_completed": self.cycle_count,
                "uptime_seconds": uptime.total_seconds(),
                "start_time": self.start_time.isoformat()
            }
        return {"status": "not_started"}
    
    def run(self):
        """Main service loop"""
        global running
        
        if not self.initialize():
            logger.error("Service initialization failed. Exiting.")
            return 1
        
        logger.info(f"Service started. Loop interval: {self.loop_interval} seconds")
        logger.info("Press Ctrl+C to stop gracefully\n")
        
        while running:
            try:
                # Process a problem
                success = self.process_problem()
                
                if not success:
                    logger.warning("Problem processing failed, but continuing...")
                
                # Log status
                status = self.get_status()
                logger.info(f"Status: {status['cycles_completed']} cycles, "
                          f"uptime: {status['uptime_seconds']:.0f}s")
                
                # Wait before next cycle
                if running:  # Check again before sleeping
                    logger.info(f"Waiting {self.loop_interval} seconds until next cycle...\n")
                    
                    # Sleep in short intervals to check for shutdown signal
                    sleep_elapsed = 0
                    while running and sleep_elapsed < self.loop_interval:
                        time.sleep(1)
                        sleep_elapsed += 1
                
            except Exception as e:
                logger.error(f"Unexpected error in service loop: {e}", exc_info=True)
                if running:
                    logger.info("Recovering... waiting 30 seconds before retry")
                    time.sleep(30)
        
        # Shutdown
        logger.info("\n" + "=" * 80)
        logger.info("NEXUS AGI SERVICE - SHUTTING DOWN")
        logger.info("=" * 80)
        status = self.get_status()
        logger.info(f"Total cycles completed: {status['cycles_completed']}")
        logger.info(f"Total uptime: {status['uptime_seconds']:.0f} seconds")
        logger.info("Service stopped gracefully")
        
        return 0

def main():
    """Entry point for the service"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Nexus AGI Continuous Service')
    parser.add_argument('--interval', type=int, default=300,
                       help='Seconds between processing cycles (default: 300)')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Create and run service
    service = NexusService(loop_interval=args.interval)
    exit_code = service.run()
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
