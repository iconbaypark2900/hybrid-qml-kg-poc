# quantum_layer/quantum_executor.py

import os
import json
import logging
import time
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
from qiskit import QuantumCircuit
try:
    from qiskit.primitives import BaseSampler
except ImportError:
    BaseSampler = None  # type: ignore[misc, assignment]
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
try:
    from qiskit_ibm_runtime import (
        QiskitRuntimeService, Sampler as RuntimeSampler,
        Session, Options
    )
except ImportError:
    QiskitRuntimeService = None  # type: ignore[misc, assignment]
    RuntimeSampler = None  # type: ignore[misc, assignment]
    Session = None  # type: ignore[misc, assignment]
    Options = None  # type: ignore[misc, assignment]
from qiskit.transpiler import PassManager
try:
    from qiskit.transpiler.passes import DynamicalDecoupling
except ImportError:
    DynamicalDecoupling = None  # type: ignore[misc, assignment]
from qiskit.circuit.library import XGate
import yaml
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class QuantumExecutor:
    """
    Unified executor for quantum circuits supporting both simulator and IBM Heron/Torino.
    """

    @staticmethod
    def gpu_available() -> bool:
        """Check if GPU-backed Aer simulation is available (cuStateVec)."""
        try:
            from qiskit_aer import AerSimulator
            AerSimulator(method='statevector', device='GPU')
            return True
        except Exception:
            return False
    
    def __init__(self, config_path: str = "config/quantum_config.yaml"):
        self.config = self._load_config(config_path)
        self.execution_mode = self.config['quantum']['execution_mode']
        self.service = None
        self.session = None
        self.noise_label = None
        self._initialize_service()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load and process configuration with environment variable substitution."""
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Substitute environment variables - FIXED to handle quotes/brackets
        def substitute_env_vars(obj):
            if isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
                var_name = obj[2:-1]
                value = os.getenv(var_name, obj)
                # Clean the value - remove any surrounding quotes or brackets
                if isinstance(value, str):
                    value = value.strip().strip('"').strip("'").strip('{').strip('}')
                return value
            elif isinstance(obj, dict):
                return {k: substitute_env_vars(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [substitute_env_vars(item) for item in obj]
            return obj
        
        return substitute_env_vars(config)

    @staticmethod
    def _resolve_ibm_runtime_identity(ibm_cfg: Dict[str, Any]) -> Tuple[str, str, Optional[str]]:
        """
        Resolve token, channel, and optional instance (CRN or hub/group/project).
        Env vars take precedence: IBM_Q_TOKEN/IBM_QUANTUM_TOKEN, IBM_QUANTUM_CHANNEL, IBM_QUANTUM_INSTANCE.
        Unset optional placeholders like ${IBM_QUANTUM_INSTANCE} become None.
        """
        token = os.environ.get("IBM_Q_TOKEN") or os.environ.get("IBM_QUANTUM_TOKEN")
        if not token:
            token = ibm_cfg.get("token")
        if isinstance(token, str):
            token = token.strip().strip('"').strip("'").strip("{").strip("}")
        channel = os.environ.get("IBM_QUANTUM_CHANNEL", "").strip()
        if not channel:
            channel = ibm_cfg.get("channel", "ibm_quantum_platform")
        if isinstance(channel, str) and channel.startswith("${"):
            channel = "ibm_quantum_platform"
        instance = os.environ.get("IBM_QUANTUM_INSTANCE", "").strip()
        if not instance:
            raw = ibm_cfg.get("instance")
            if isinstance(raw, str):
                raw = raw.strip()
                if raw and not (raw.startswith("${") and raw.endswith("}")):
                    instance = raw
        if not instance:
            instance = None
        return token or "", channel, instance

    def _initialize_service(self):
        """Initialize IBM Quantum Runtime service if needed."""
        if self.execution_mode == "heron" or self.execution_mode == "auto":
            if QiskitRuntimeService is None:
                logger.warning("⚠️  qiskit_ibm_runtime not installed. Heron mode unavailable.")
                if self.execution_mode == "heron":
                    logger.info("🔄 Falling back to simulator mode")
                    self.execution_mode = "simulator"
                return
            try:
                ibm_cfg = self.config.get("quantum", {}).get("ibm_quantum", {})
                token, channel, instance = self._resolve_ibm_runtime_identity(ibm_cfg)
                
                if token and token != "your_actual_token_here":
                    kwargs: Dict[str, Any] = {"channel": channel, "token": token}
                    if instance:
                        kwargs["instance"] = instance
                    self.service = QiskitRuntimeService(**kwargs)
                    if instance:
                        logger.info("✅ Connected to IBM Quantum (%s, instance set)", channel)
                    else:
                        logger.info("✅ Connected to IBM Quantum (%s)", channel)
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

    def _get_simulator_sampler(self):
        """Get simulator sampler - fixed to return proper tuple"""
        from qiskit.primitives import Sampler
        try:
            from qiskit.primitives import StatevectorSampler
            return StatevectorSampler(), "simulator_statevector"
        except Exception:
            return Sampler(), "simulator"

    def _get_heron_sampler(self) -> Tuple[RuntimeSampler, str]:
        """Get IBM Heron/Torino sampler with error mitigation."""
        if not self.service:
            raise RuntimeError("IBM Quantum service not available")
        
        heron_config = self.config['quantum']['heron']
        backend_name = heron_config['backend']
        
        try:
            backend = self.service.backend(backend_name)
            logger.info(f"🚀 Using IBM Quantum backend: {backend_name}")
            
            # Configure options - FIXED for new API (removed unsupported options)
            options = Options()
            # New API uses different structure - check if execution exists
            if hasattr(options, 'execution'):
                options.execution.shots = heron_config['shots']
            # max_execution_time may not be in Options anymore
            if hasattr(options, 'max_execution_time'):
                options.max_execution_time = heron_config['max_runtime_minutes'] * 60
            
            # Note: optimization_level and resilience_level were removed in newer API versions
            # These are now handled at the transpiler level, not in Options
            
            # Create session for cost efficiency
            self.session = Session(backend=backend)
            sampler = RuntimeSampler(session=self.session, options=options)
            
            return sampler, backend_name
            
        except Exception as e:
            logger.error(f"❌ Backend {backend_name} unavailable: {e}")
            raise

    def get_sampler(self):
        """Get sampler - fixed to always return tuple"""
        # Prefer exact sims locally (statevector if available)
        if self.execution_mode in ("simulator", "auto", "statevector", "simulator_statevector"):
            return self._get_simulator_sampler()

        if self.execution_mode == "heron":
            try:
                return self._get_heron_sampler()
            except Exception as e:
                logger.warning(f"Heron unavailable, using simulator: {e}")
                return self._get_simulator_sampler()

        # safe default
        return self._get_simulator_sampler()
    
    def optimize_circuit(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Apply circuit optimization passes."""
        # Check if circuit optimization is enabled (add to config if needed)
        if not self.config['quantum'].get('circuit_optimization', True):
            return circuit
        
        optimized = circuit.copy()
        
        # Add dynamical decoupling for Heron (when pass is available)
        if (DynamicalDecoupling is not None and self.execution_mode == "heron" and
            self.config['quantum']['heron']['use_dynamical_decoupling']):
            try:
                backend = self.service.backend(self.config['quantum']['heron']['backend']) if self.service else None
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