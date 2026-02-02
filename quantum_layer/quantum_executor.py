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
from qiskit.transpiler.passes import DynamicalDecoupling
from qiskit.circuit.library import XGate
import yaml
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class QuantumExecutor:
    """
    Unified executor for quantum circuits supporting both simulator and IBM Heron/Torino.
    """
    
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
                token = self.config['quantum']['ibm_quantum']['token']
                instance = self.config['quantum']['ibm_quantum'].get('instance')
                channel = self.config['quantum']['ibm_quantum'].get('channel', 'ibm_quantum_platform')
                
                # Additional cleaning of token
                if isinstance(token, str):
                    token = token.strip().strip('"').strip("'").strip('{').strip('}')
                
                if token and token != "your_actual_token_here":
                    # Use IBM Quantum Platform (simpler)
                    self.service = QiskitRuntimeService(
                        channel="ibm_quantum_platform",
                        token=token
                    )
                    logger.info("✅ Connected to IBM Quantum Platform")
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

    def _load_noise_model(self) -> Tuple[Optional[NoiseModel], Optional[str]]:
        """Load a noise model from config if configured."""
        sim_config = self.config.get('quantum', {}).get('simulator', {})
        noise_spec = sim_config.get('noise_model')
        if not noise_spec:
            return None, None

        if isinstance(noise_spec, dict):
            try:
                return NoiseModel.from_dict(noise_spec), "custom_dict"
            except Exception as e:
                logger.warning(f"⚠️  Failed to load noise model from dict: {e}")
                return None, None

        if isinstance(noise_spec, str):
            spec = noise_spec.strip()
            # Built-in depolarizing model shortcut: "depolarizing:0.01"
            if spec.startswith("depolarizing:"):
                try:
                    from qiskit_aer.noise import depolarizing_error
                    prob = float(spec.split(":", 1)[1])
                    noise_model = NoiseModel()
                    one_qubit_gates = ["x", "y", "z", "h", "s", "t", "sx", "rz", "rx", "ry"]
                    two_qubit_gates = ["cx", "cz", "swap", "ecr"]
                    noise_model.add_all_qubit_quantum_error(depolarizing_error(prob, 1), one_qubit_gates)
                    noise_model.add_all_qubit_quantum_error(depolarizing_error(prob, 2), two_qubit_gates)
                    return noise_model, spec
                except Exception as e:
                    logger.warning(f"⚠️  Failed to build depolarizing noise model: {e}")
                    return None, None

            # Load from JSON file
            if Path(spec).exists():
                try:
                    with open(spec, "r") as f:
                        data = json.load(f)
                    return NoiseModel.from_dict(data), spec
                except Exception as e:
                    logger.warning(f"⚠️  Failed to load noise model from {spec}: {e}")
                    return None, None

            # Load from IBM backend if available: "ibm_backend:ibm_torino"
            if spec.startswith("ibm_backend:"):
                backend_name = spec.split(":", 1)[1]
                if self.service:
                    try:
                        backend = self.service.backend(backend_name)
                        return NoiseModel.from_backend(backend), spec
                    except Exception as e:
                        logger.warning(f"⚠️  Failed to load noise model from backend {backend_name}: {e}")
                        return None, None
                logger.warning("⚠️  IBM Quantum service unavailable; cannot load backend noise model.")
                return None, None

            logger.warning(f"⚠️  Unrecognized noise_model spec: {spec}")
            return None, None

        logger.warning(f"⚠️  Unsupported noise_model type: {type(noise_spec)}")
        return None, None

    def _get_simulator_sampler(self):
        """Get simulator sampler - fixed to return proper tuple"""
        sim_config = self.config.get('quantum', {}).get('simulator', {})
        shots = int(sim_config.get("shots", 1024) or 1024)

        noise_model, noise_label = self._load_noise_model()
        if noise_model is not None:
            try:
                # Qiskit Machine Learning expects a BaseSamplerV2-compatible primitive.
                from qiskit_aer.primitives import SamplerV2 as AerSamplerV2
                self.noise_label = noise_label
                return (
                    AerSamplerV2(
                        default_shots=shots,
                        options={"backend_options": {"noise_model": noise_model}},
                    ),
                    "simulator_noisy",
                )
            except Exception as e:
                logger.warning(f"⚠️  Failed to initialize noisy simulator, falling back to ideal: {e}")

        try:
            from qiskit.primitives import StatevectorSampler
            return StatevectorSampler(), "simulator_statevector"
        except Exception:
            pass
        try:
            from qiskit_aer.primitives import SamplerV2 as AerSamplerV2
            return AerSamplerV2(default_shots=shots), "simulator"
        except Exception:
            pass
        try:
            from qiskit.primitives import Sampler
            return Sampler(), "simulator"
        except Exception:
            from qiskit_aer.primitives import Sampler
            return Sampler(), "simulator"

    def _get_heron_sampler(self) -> Tuple[Any, str]:
        """Get IBM Heron/Torino sampler with error mitigation."""
        if RuntimeSampler is None or Session is None or Options is None:
            raise RuntimeError(
                "qiskit_ibm_runtime is required for Heron mode. "
                "Install with: pip install qiskit_ibm_runtime"
            )
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

            # Best-effort: apply resilience/optimization knobs if exposed by this Options version.
            # (IBM runtime APIs shift over time; we keep this defensive.)
            opt_level = heron_config.get("optimization_level")
            res_level = heron_config.get("resilience_level")
            try:
                if opt_level is not None and hasattr(options, "optimization_level"):
                    options.optimization_level = int(opt_level)
                if res_level is not None and hasattr(options, "resilience_level"):
                    options.resilience_level = int(res_level)
            except Exception as e:
                logger.warning(f"⚠️  Could not apply optimization/resilience levels: {e}")
            # Some versions expose nested resilience/execution option blocks
            try:
                if hasattr(options, "resilience") and res_level is not None and hasattr(options.resilience, "level"):
                    options.resilience.level = int(res_level)
            except Exception:
                pass
            
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

    def get_execution_metadata(self) -> Dict[str, Optional[str]]:
        """Return execution metadata for logging."""
        execution_mode = self.execution_mode
        sim_cfg = self.config.get('quantum', {}).get('simulator', {})
        noise_model = sim_cfg.get('noise_model')
        heron_cfg = self.config.get('quantum', {}).get('heron', {}) if isinstance(self.config.get('quantum', {}).get('heron', {}), dict) else {}

        if execution_mode == "heron":
            backend_label = self.config.get('quantum', {}).get('heron', {}).get('backend')
        elif execution_mode in ("simulator", "auto", "statevector", "simulator_statevector"):
            backend_label = "simulator_noisy" if noise_model else "simulator"
        else:
            backend_label = execution_mode

        return {
            "execution_mode": execution_mode,
            "noise_model": str(noise_model) if noise_model is not None else None,
            "backend_label": backend_label,
            "mitigation_resilience_level": str(heron_cfg.get("resilience_level")) if execution_mode == "heron" and heron_cfg.get("resilience_level") is not None else None,
            "mitigation_optimization_level": str(heron_cfg.get("optimization_level")) if execution_mode == "heron" and heron_cfg.get("optimization_level") is not None else None,
            "mitigation_dynamical_decoupling": str(heron_cfg.get("use_dynamical_decoupling")) if execution_mode == "heron" and heron_cfg.get("use_dynamical_decoupling") is not None else None,
        }
    
    def optimize_circuit(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Apply circuit optimization passes."""
        # Check if circuit optimization is enabled (add to config if needed)
        if not self.config['quantum'].get('circuit_optimization', True):
            return circuit
        
        optimized = circuit.copy()
        
        # Add dynamical decoupling for Heron
        if (self.execution_mode == "heron" and 
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