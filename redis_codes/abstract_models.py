import asyncio
import json
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union


def get_priority_name(queue_name, priority):
    if priority == 0:
        return f"{queue_name}:high"
    else:
        return f"{queue_name}:low"

def transform_priority_name(queue_name:dict, priority:str):
    priority_level = priority.split(":")[-1]
    return queue_name[priority_level]

def get_priority_number(priority:str):
    if priority.endswith("high"):
        return 1
    elif priority.endswith("low"):
        return 2
    elif priority.endswith("triger"):
        return 0
    else:
        return -1

def get_high_low(inputs:List[str]):
    high_priority_channle_name = ""
    low_priority_channle_name = ""
    for i in inputs:
        if i.endswith(":high"):
            high_priority_channle_name = i
        elif i.endswith(":low"):
            low_priority_channle_name = i
    return {"high":high_priority_channle_name, "low": low_priority_channle_name}
  

@dataclass
class ChannelNames:
    input_channel:Union[str, List[str]]
    output_channel:Union[str, List[str]]
    channel_names:List[str] = field(default_factory=list)
    
    def get_all_channels(self):
        if isinstance(self.input_channel, str):
            input_channel = [self.input_channel]
        else:
            input_channel = self.input_channel
        if isinstance(self.output_channel, str):
            output_channel = [self.output_channel]
        else:
            output_channel = self.output_channel
        return input_channel + output_channel + self.channel_names
    

@dataclass
class Features:
    sid: str
    agent_name:str
    priority:str
    created_at: float
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()

    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.timeout

    def to_json(self) -> str:
        """Convert instance to JSON string"""
        return json.dumps(asdict(self))
    
    @classmethod
    def from_json(cls, json_str: str):
        """Create instance from JSON string"""
        data = json.loads(json_str)
        return cls(**data)
    
    def refresh_time(self):
        self.created_at = time.time()

    
# @dataclass
# class SessionStatus:
#     sid:str
#     status:str
#     timeout:float
#     created_at: float=None
    
#     def __post_init__(self):
#         if self.created_at is None:
#             self.created_at = time.time()
            
#     def is_expired(self) -> bool:
#         return time.time() - self.created_at > self.timeout
    
#     def to_json(self) -> str:
#         """Convert instance to JSON string"""
#         return json.dumps(asdict(self))        
    
#     @classmethod
#     def from_json(cls, json_str: str):
#         """Create instance from JSON string"""
#         data = json.loads(json_str)
#         return cls(**data)
    
#     def refresh_time(self):
#         self.created_at = time.time()


@dataclass
class SessionStatus:
    ACTIVE = "active"
    INTERRUPT = "interrupt"
    STOP = "stop"


def go_next_service(current_stage_name:str, service_names:Optional[List[str]], channels_steps:Optional[Dict[str, List[str]]], last_channel:str, prioriry:str) -> bool:
    """
    Advance the session to the next service in the pipeline.
    Returns True if successfully advanced, False if already at the end.
    """
    if current_stage_name == "start":
        # First real stage is the first service
        if service_names:
            current_stage_name = service_names[0]
            if prioriry not in channels_steps[current_stage_name]:
                return None
            next_channel = f"{current_stage_name}:{prioriry}"
            return next_channel
        else:
            return None

    try:
        idx = service_names.index(current_stage_name)
    except ValueError:
        # Current stage not in pipeline – cannot advance
        return None
    if idx + 1 < len(service_names):
        # Move to next service
        next_service = service_names[idx + 1]
        current_stage_name = next_service
        if prioriry not in channels_steps[current_stage_name]:
            return None
        
        next_channel = f"{next_service}:{prioriry}"
        return next_channel
    return last_channel
    
class AgentSessions:
    def __init__(
        self,
        sid: str,
        agent_name: str,
        service_names: Optional[List[str]],
        channels_steps: Optional[Dict[str, List[str]]],
        owner_id:str,
        kb_id:List[str]=[None],
        kb_limit:int=5,
        status: SessionStatus = SessionStatus.ACTIVE,
        timeout: float = 30.0,
        first_channel:str=None,
        last_channel:str=None,
        created_at: Optional[float] = None,
    ):
        self.sid = sid
        self.status = status
        self.timeout = timeout
        self.created_at = created_at if created_at is not None else time.time()
        self.agent_name = agent_name
        self.first_channel = first_channel
        self.last_channel = last_channel
        self.service_names = service_names or []
        self.channels_steps = OrderedDict(channels_steps or {})
        self.owner_id = owner_id
        self.kb_id = kb_id
        self.kb_limit = kb_limit

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
            
    def refresh_time(self) -> None:
        """Update the creation timestamp to extend session lifetime."""
        self.created_at = time.time()

    def is_expired(self) -> bool:
        """Check if the session has exceeded its timeout."""
        return (time.time() - self.created_at) > self.timeout        

    def to_json(self) -> str:
        """Serialize session state to JSON string."""
        data = {
            "sid": self.sid,
            "status": self.status,
            "timeout": self.timeout,
            "created_at": self.created_at,
            "agent_name": self.agent_name,
            "first_channel": self.first_channel,
            "last_channel": self.last_channel,
            "service_names": self.service_names,
            "owner_id" : self.owner_id,
            "kb_id" : self.kb_id,
            "kb_limit" : self.kb_limit,
            "channels_steps": dict(self.channels_steps),  # OrderedDict → dict for JSON
        }
        return json.dumps(data)

    @classmethod
    def from_json(cls, json_str: str) -> "AgentSessions":
        """Deserialize session state from JSON string."""
        data = json.loads(json_str)
        # Reconstruct OrderedDict for channels_steps
        channels_steps = OrderedDict(data.get("channels_steps", {}))
        return cls(
            sid=data["sid"],
            status=data["status"],
            timeout=data["timeout"],
            created_at=data["created_at"],
            agent_name=data["agent_name"],
            first_channel=data["first_channel"],
            last_channel=data["last_channel"],
            service_names=data["service_names"],
            owner_id=data["owner_id"],
            kb_id=data["kb_id"],
            kb_limit=data["kb_limit"],
            channels_steps=channels_steps,
        )

    def __repr__(self) -> str:
        return (f"AgentSessions(sid={self.sid}, agent={self.agent_name}, status={self.status}, create_at={self.created_at}, first_channel={self.first_channel}, last_channel={self.last_channel}, owner_id: {self.owner_id}") 
    

def get_all_channels(req:AgentSessions):
    middle_channels = []
    for service in req.service_names:
        if service in req.channels_steps:
            for priority in req.channels_steps[service]:
                middle_channels.append(f"{service}:{priority}")
    return middle_channels + [req.first_channel] + [req.last_channel]
     
@dataclass
class AudioFeatures(Features):
    audio:bytes
    sample_rate:int
    
    
@dataclass
class TextFeatures(Features):
    text:str

@dataclass
class RAGFeatures(TextFeatures):
    owner_id:str
    kb_id:List[str]
    kb_limit:int

class AbstractQueueManager(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    async def initialize(self):
        pass
    
    @abstractmethod
    async def is_session_active(self, sid: str) -> bool:
        """Check if a session is active"""
        pass
    
    async def cleanup_session_requests(self, sid: str):
        """Remove any queued requests for the sid """
        pass

class AbstractQueueManagerServer(AbstractQueueManager):
    @abstractmethod
    async def get_data_batch(self):
        """Retrieve a batch of audio data from the processing queue."""
        pass
    
    async def is_session_interrupt(self, sid:str):
        """interrupt status for the spacial session id"""
        pass
    
    @abstractmethod
    async def push_result(self):
        """Publish the result of the speech-to-text processing."""
        pass
    
class AbstractQueueManagerClient(AbstractQueueManager):
    
    @abstractmethod
    async def start_session(self, sid: str):
        """Start a new client session"""
        pass

    @abstractmethod
    async def stop_session(self, sid: str):
        """Stop a specific client session"""
        pass
    
    @abstractmethod
    async def submit_data_request(self):
        """Submit an audio request to the processing queue."""
        pass
    
    @abstractmethod
    async def listen_for_result(self):
        """Listen for the result of the speech-to-text processing."""
        pass
    
    @abstractmethod
    async def close(self):
        """Close the queue manager, releasing any resources."""
        pass
        
class AbstractAsyncModelInference(ABC):
    """
    Abstract base class for asynchronous model inference with dynamic batching.
    Subclasses must implement model-specific logic.
    """

    def __init__(self, max_worker: int = 4):
        self.thread_pool = ThreadPoolExecutor(max_workers=max_worker)
        self.stats = {
            'total_batches': 0,
            'total_requests': 0,
            'avg_batch_size': 0,
            'avg_inference_time': 0
        }

    async def process_batch(self, batch: List[TextFeatures]) -> Dict[str, Any]:
        """Process a batch of requests asynchronously (final method)."""
        pass

    async def _prepare_batch_inputs(self, batch: List[Features]) -> Any:
        """Prepare inputs for inference. Return format is implementation-defined."""
        pass

    async def _run_async_model_inference(self, prepared_inputs: Any) -> Any:
        """Run async inference (e.g., with vLLM). Must be implemented."""
        pass
    
    def _run_model_inference(self, prepared_inputs: Any) -> Any:
        """Run synchronous inference in thread pool. Must not be async."""
        pass

    async def _process_batch_outputs(self, outputs: Any, batch: List[Features]) -> Dict[str, Any]:
        """Map raw outputs back to request IDs and format results."""
        pass

    async def _handle_batch_error(self, batch: List[Features], error: Exception) -> Dict[str, Any]:
        """Handle errors during batch processing (can be overridden)."""
        error_results = {}
        for request in batch:
            error_results[request.sid] = {
                'result': None,
                'error': str(error)
            }
        return error_results

    def _update_stats(self, batch_size: int, processing_time: float):
        """Update internal statistics (final method)."""
        self.stats['total_batches'] += 1
        self.stats['total_requests'] += batch_size
        self.stats['avg_batch_size'] = (
            self.stats['avg_batch_size'] * (self.stats['total_batches'] - 1) + batch_size
        ) / self.stats['total_batches']
        self.stats['avg_inference_time'] = (
            self.stats['avg_inference_time'] * (self.stats['total_batches'] - 1) + processing_time
        ) / self.stats['total_batches']
  
class AbstractInference(ABC):
    """
    Abstract base class for a complete async inference service with dynamic batching,
    queue management, and result publishing.
    """
    @abstractmethod    
    def __init__(self):
        self.is_running = False
        self.processing_task: Optional[asyncio.Task] = None
        self.active_sessions: Dict[str, bool] = {}
        self.session_locks: Dict[str, asyncio.Lock] = {}
    
    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        """Stop the inference service gracefully."""
        pass
    
    async def is_session_active(self, sid: str) -> bool:
        """Check if a session is active"""
        pass    
    
    @abstractmethod
    async def _initialize_components(self) -> None:
        """Initialize queue manager, inference engine, etc."""
        pass
   
class AbstractInferenceServer(AbstractInference):
    """
    Abstract base class for a complete async inference service with dynamic batching,
    queue management, and result publishing.
    """
    # async def start(self) -> None:
    #     """Start the inference service."""
    #     await self._initialize_components()
    #     self.is_running = True
    #     self.processing_task = asyncio.create_task(self._process_batches_loop())
        
    
    @abstractmethod
    async def _process_batches_loop(self) -> None:
        """
        Main loop that:
        - Fetches a batch of requests (with dynamic batching logic)
        - Runs inference
        - Publishes results
        - Updates metrics
        """
        pass

class AbstractInferenceClient(AbstractInference):
    """
    Abstract base class for a complete async inference service with dynamic batching,
    queue management, and result publishing.
    """
    
    async def start_session(self, sid: str):
        """Start a new client session"""
        pass

    
    async def stop_session(self, sid: str):
        """Stop a specific client session"""
        pass
    
    
    async def start(self) -> None:
        """Start the inference service."""
        await self._initialize_components()
        self.is_running = True
    
    
    async def stop(self) -> None:
        """Stop the inference service gracefully."""
        self.is_running = False
        if self.processing_task:
            await self.processing_task
        await self._cleanup_components()
    
    
    async def _cleanup_components(self) -> None:
        """Clean up resources (close connections, executors, etc.)."""
        pass

    
    async def predict(
        self,
        input_data: Any,
        sid: str,
        priority: int = 1,
        timeout: float = 30.0
    ) -> Any:
        """
        Public API for submitting a prediction request and awaiting its result.
        
        Parameters:
            input_data: The input payload (e.g., text string).
            sid: Unique session/request ID.
            priority: Request priority (e.g., 0=trigger, 1=high, 2=low).
            timeout: Max time to wait for result (seconds).

        Returns:
            The inference result (e.g., audio array and sample rate).

        Raises:
            Exception: If inference fails or times out.
        """
        pass

class DynamicBatchManager:
    """
    Implements dynamic batching strategies inspired by NVIDIA Triton :cite[2]
    """
    
    def __init__(
        self,
        max_batch_size: int = 16,
        max_wait_time: float = 0.1,
        preferred_batch_sizes: List[int] = None
    ):
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.preferred_batch_sizes = preferred_batch_sizes or [4, 8, 16]
        
        # Metrics
        self.batch_sizes = []
        self.avg_processing_time = 0.0
        
    def should_process_batch(self, current_batch: List[Features], batch_start_time: float) -> bool:
        """Determine if current batch should be processed now"""
        current_size = len(current_batch)
        current_time = time.time()
        batch_age = current_time - batch_start_time
        
        # Check if we reached max batch size :cite[7]
        if current_size >= self.max_batch_size:
            return True
        
        # Check if we reached a preferred batch size :cite[2]
        if current_size in self.preferred_batch_sizes:
            return True
        
        # Check if max wait time reached (delayed batching) :cite[2]
        if batch_age >= self.max_wait_time and current_size > 0:
            return True
        
        # Check for urgent requests (approaching timeout)
        if self._has_urgent_requests(current_batch):
            return True
            
        return False
    
    def _has_urgent_requests(self, batch: List[Features]) -> bool:
        """Check if any requests are approaching timeout"""
        current_time = time.time()
        for request in batch:
            time_until_timeout = request.timeout - (current_time - request.created_at)
            if time_until_timeout < self.max_wait_time:
                return True
        return False
    
    def update_metrics(self, batch_size: int, processing_time: float):
        """Update batch processing metrics"""
        self.batch_sizes.append(batch_size)
        if len(self.batch_sizes) > 100:  # Keep last 100 batches
            self.batch_sizes.pop(0)
        
        # Update running average
        self.avg_processing_time = (
            self.avg_processing_time * (len(self.batch_sizes) - 1) + processing_time
        ) / len(self.batch_sizes)
                       
    