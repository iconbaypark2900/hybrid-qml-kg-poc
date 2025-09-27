# quantum_layer/quantum_executor.py

import os
import logging
import time
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
from qiskit import QuantumCircuit
from qiskit.primitives import BaseSampler
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import (
    QiskitRuntimeService, Sampler as RuntimeSampler, 
    Session, Options
)
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import DynamicalDecoupling
from qiskit.circuit.library import XGate
import yaml
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class QuantumExecutor:
    """
    Unified executor for quantum circuits supporting both simulator and IBM Heron.
    """
    
    def __init__(self, config_path: str = "config/quantum_config.yaml"):
        self.config = self._load_config(config_path)
        self.execution_mode = self.config['quantum']['execution_mode']
        self.service = None
        self.session = None
        self._initialize_service()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load and process configuration with environment variable substitution."""
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Substitute environment variables
        def substitute_env_vars(obj):
            if isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
                var_name = obj[2:-1]
                return os.getenv(var_name, obj)
            elif isinstance(obj, dict):
                return {k: substitute_env_vars(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [substitute_env_vars(item) for item in obj]
            return obj
        
        return substitute_env_vars(config)
    
    def _initialize_service(self):
        """Initialize IBM Quantum Runtime service if needed."""
        if self.execution_mode == "heron" or self.execution_mode == "auto":
            try:
                token = self.config['quantum']['ibm_quantum']['token']
                if token and token != "your_actual_token_here":
                    self.service = QiskitRuntimeService(
                        channel="ibm_quantum",
                        token=token,
                        instance=self.config['quantum']['ibm_quantum']['instance']
                    )
                    logger.info("✅ Connected to IBM Quantum Runtime Service")
                else:
                    logger.warning("⚠️  IBM Quantum token not configured. Heron mode unavailable.")
                    if self.execution_mode == "heron":
                        logger.info("🔄 Falling back to simulator mode")
                        self.execution_mode = "simulator"
            except Exception as e:
                logger.error(f"❌ Failed to connect to IBM Quantum: {e}")
                if self.execution_mode == "heron":
                    logger.info("🔄 Falling back to simulator mode")
                    self.execution_mode = "simulator"
    
    def _get_simulator_sampler(self) -> Tuple[BaseSampler, Optional[str]]:
        """Get simulator sampler with optional noise model."""
        sim_config = self.config['quantum']['simulator']
        
        if sim_config.get('noise_model') == "ibm_heron_noise" and self.service:
            # Get realistic Heron noise model
            try:
                backend = self.service.backend("ibm_heron")
                noise_model = NoiseModel.from_backend(backend)
                simulator = AerSimulator(noise_model=noise_model)
                logger.info("🧪 Using Heron noise model in simulator")
            except Exception as e:
                logger.warning(f"⚠️  Could not load Heron noise model: {e}")
                simulator = AerSimulator()
        else:
            simulator = AerSimulator()
        
        # Configure shots
        from qiskit.primitives import Sampler as LocalSampler
        sampler = LocalSampler(backend=simulator, options={"shots": sim_config['shots']})
        return sampler, "simulator"
    
    def _get_heron_sampler(self) -> Tuple[RuntimeSampler, str]:
        """Get IBM Heron sampler with error mitigation."""
        if not self.service:
            raise RuntimeError("IBM Quantum service not available")
        
        heron_config = self.config['quantum']['heron']
        backend_name = self.config['quantum']['ibm_quantum']['backend']
        
        try:
            backend = self.service.backend(backend_name)
            logger.info(f"🚀 Using IBM Quantum backend: {backend_name}")
            
            # Configure options
            options = Options()
            options.resilience_level = heron_config['resilience_level']
            options.optimization_level = heron_config['optimization_level']
            options.execution.shots = heron_config['shots']
            options.max_execution_time = heron_config['max_runtime_minutes'] * 60
            
            # Create session for cost efficiency
            self.session = Session(backend=backend)
            sampler = RuntimeSampler(session=self.session, options=options)
            
            return sampler, backend_name
            
        except Exception as e:
            logger.error(f"❌ Heron backend unavailable: {e}")
            raise
    
    def get_sampler(self) -> Tuple[BaseSampler, str]:
        """Get appropriate sampler based on execution mode."""
        if self.execution_mode == "simulator":
            return self._get_simulator_sampler()
        elif self.execution_mode == "heron":
            return self._get_heron_sampler()
        elif self.execution_mode == "auto":
            # Try Heron first, fallback to simulator
            try:
                return self._get_heron_sampler()
            except Exception as e:
                logger.warning(f"🔄 Heron unavailable, using simulator: {e}")
                return self._get_simulator_sampler()
        else:
            raise ValueError(f"Unknown execution mode: {self.execution_mode}")
    
    def optimize_circuit(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Apply circuit optimization passes."""
        if not self.config['quantum']['circuit_optimization']:
            return circuit
        
        optimized = circuit.copy()
        
        # Add dynamical decoupling for Heron
        if (self.execution_mode == "heron" and 
            self.config['quantum']['heron']['use_dynamical_decoupling']):
            try:
                backend = self.service.backend("ibm_heron") if self.service else None
                if backend:
                    dd_pass = DynamicalDecoupling(backend.target.dt, [XGate()])
                    pm = PassManager([dd_pass])
                    optimized = pm.run(optimized)
                    logger.debug("✨ Applied dynamical decoupling")
            except Exception as e:
                logger.warning(f"⚠️  Could not apply dynamical decoupling: {e}")
        
        return optimized
    
    def estimate_cost(self, circuit: QuantumCircuit, shots: int = None) -> Dict[str, Any]:
        """Estimate execution cost and time."""
        if shots is None:
            shots = (self.config['quantum']['heron']['shots'] 
                    if self.execution_mode == "heron" 
                    else self.config['quantum']['simulator']['shots'])
        
        gate_count = circuit.size()
        qubit_count = circuit.num_qubits
        
        if self.execution_mode == "heron":
            # Heron cost estimation (adjust based on current IBM pricing)
            credits_per_shot = 0.0005  # Approximate
            estimated_credits = shots * credits_per_shot
            estimated_queue_time = 30  # minutes (highly variable)
            
            return {
                "mode": "heron",
                "estimated_credits": estimated_credits,
                "estimated_queue_minutes": estimated_queue_time,
                "gate_count": gate_count,
                "qubit_count": qubit_count,
                "shots": shots
            }
        else:
            # Simulator is free
            return {
                "mode": "simulator",
                "estimated_credits": 0,
                "estimated_runtime_seconds": gate_count * 0.001,  # Rough estimate
                "gate_count": gate_count,
                "qubit_count": qubit_count,
                "shots": shots
            }
    
    def close_session(self):
        """Close IBM Quantum session to avoid lingering costs."""
        if self.session:
            try:
                self.session.close()
                logger.info("🔒 Closed IBM Quantum session")
            except Exception as e:
                logger.warning(f"⚠️  Could not close session: {e}")
    
    def __del__(self):
        """Cleanup on destruction."""
        self.close_session()