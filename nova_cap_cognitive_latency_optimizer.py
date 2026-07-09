class CognitiveLatencyOptimizer:
    def __init__(self):
        self.state = {}

    def status(self) -> dict:
        return self.state

    def analyze_latency(self, module_name: str, latency: float):
        if not module_name or not isinstance(latency, (int, float)) or latency < 0:
            raise ValueError("Invalid module name or latency value")
        try:
            self.state[module_name] = latency
        except Exception as e:
            raise  # re-raise exception instead of ignoring it

    def optimize_latency(self, module_name: str):
        try:
            if module_name and module_name in self.state:
                latency = self.state[module_name]
                if latency >= 0:
                    # Simple optimization strategy: reduce latency by 10%
                    self.state[module_name] = max(0, latency * 0.9)
                else:
                    raise ValueError("Latency value cannot be negative")
        except Exception as e:
            raise  # re-raise exception instead of ignoring it

    def get_latency(self, module_name: str) -> float:
        try:
            if module_name:
                return self.state[module_name]
            else:
                raise ValueError("Module name cannot be empty or None")
        except KeyError:
            raise  # re-raise KeyError instead of returning 0.0
        except Exception as e:
            raise  # re-raise exception instead of ignoring it

    def remove_module(self, module_name: str):
        try:
            if module_name:
                if module_name in self.state:
                    del self.state[module_name]
                else:
                    raise ValueError("Module not found")
            else:
                raise ValueError("Module name cannot be empty or None")
        except Exception as e:
            raise  # re-raise exception instead of ignoring it

# Usage: obj = CognitiveLatencyOptimizer()
