import atexit
import json
import shutil
import subprocess
import threading
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from facefusion import logger, state_manager

HEARTBEAT_INTERVAL = 15
# FaceFusion keeps a single global state, so a node serves one session at a time — more would clobber each other
MAX_SESSIONS = 1
WORKFLOWS = [ 'image-to-image', 'image-to-video' ]

__node_id__ : Optional[str] = None
__stop_event__ = threading.Event()


def read_value(value_and_unit : Optional[Dict[str, Any]]) -> Optional[int]:
	return value_and_unit.get('value') if value_and_unit else None


def read_nvidia_gpu() -> Optional[Dict[str, Any]]:
	if not shutil.which('nvidia-smi'):
		return None

	try:
		output = subprocess.check_output(
		[
			'nvidia-smi',
			'--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu',
			'--format=csv,noheader,nounits'
		], timeout = 5).decode().strip()
		name, utilization, memory_used, memory_total, temperature = [ value.strip() for value in output.splitlines()[0].split(',') ]

		return\
		{
			'name': name,
			'utilization': int(utilization),
			'memory_used': round(int(memory_used) / 1024, 1),
			'memory_total': round(int(memory_total) / 1024, 1),
			'memory_unit': 'gb',
			'temperature': int(temperature)
		}
	except (OSError, ValueError, subprocess.SubprocessError):
		return None


def collect_metrics() -> Optional[Dict[str, Any]]:
	try:
		from facefusion.system import get_metrics_set

		metrics_set = get_metrics_set()
		graphic_devices = metrics_set.get('graphic_devices') or []
		processor = metrics_set.get('processor') or {}
		memory = metrics_set.get('memory') or {}
		metrics : Dict[str, Any] =\
		{
			'cpu':
			{
				'cores': read_value(processor.get('cores')),
				'utilization': read_value(processor.get('utilization'))
			},
			'memory':
			{
				'total': read_value(memory.get('total')),
				'unit': (memory.get('total') or {}).get('unit'),
				'utilization': read_value(memory.get('utilization'))
			}
		}

		if graphic_devices:
			graphic_device = graphic_devices[0]
			metrics['gpu'] =\
			{
				'name': (graphic_device.get('product') or {}).get('name'),
				'utilization': read_value((graphic_device.get('utilization') or {}).get('gpu')),
				'memory_used': read_value((graphic_device.get('memory') or {}).get('used')),
				'memory_total': read_value((graphic_device.get('memory') or {}).get('total')),
				'memory_unit': ((graphic_device.get('memory') or {}).get('total') or {}).get('unit'),
				'temperature': read_value((graphic_device.get('temperature') or {}).get('gpu'))
			}
		else:
			# no CUDA/ROCm provider active — fall back to nvidia-smi so the real GPU still shows
			nvidia_gpu = read_nvidia_gpu()

			if nvidia_gpu:
				metrics['gpu'] = nvidia_gpu

		return metrics
	except Exception:
		return None


def resolve_broker_url() -> str:
	api_domain = state_manager.get_item('api_domain')

	if '://' in api_domain:
		return api_domain.rstrip('/')
	return 'http://' + api_domain.rstrip('/')


def resolve_node_endpoint() -> str:
	api_host = state_manager.get_item('api_host')
	api_port = state_manager.get_item('api_port')

	if api_host in [ '0.0.0.0', '' ]:
		api_host = '127.0.0.1'
	return 'http://{}:{}'.format(api_host, api_port)


def request_broker(method : str, path : str, payload : Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
	url = resolve_broker_url() + path
	data = json.dumps(payload).encode() if payload is not None else None
	request = urllib.request.Request(url, data = data, method = method, headers = { 'Content-Type': 'application/json' })

	with urllib.request.urlopen(request, timeout = 10) as response:
		content = response.read()
		return json.loads(content) if content else {}


def register_with_broker() -> Optional[str]:
	payload : Dict[str, Any] =\
	{
		'endpoint': resolve_node_endpoint(),
		'visibility': 'public',
		'max_sessions': MAX_SESSIONS,
		'processors': state_manager.get_item('processors'),
		'workflows': WORKFLOWS,
		'metrics': collect_metrics()
	}
	content = request_broker('POST', '/swarm/nodes', payload)
	return content.get('node_id')


def join_swarm() -> bool:
	global __node_id__

	try:
		__node_id__ = register_with_broker()
	except (urllib.error.URLError, OSError) as exception:
		logger.error(translate_error(exception), __name__)
		return False

	heartbeat_thread = threading.Thread(target = run_heartbeat, daemon = True)
	heartbeat_thread.start()
	atexit.register(leave_swarm)
	logger.info('Joined the swarm at {} as node {}'.format(resolve_broker_url(), __node_id__), __name__)
	return True


def run_heartbeat() -> None:
	global __node_id__

	while not __stop_event__.wait(HEARTBEAT_INTERVAL):
		try:
			request_broker('POST', '/swarm/nodes/{}/heartbeat'.format(__node_id__), { 'metrics': collect_metrics() })
		except urllib.error.HTTPError as exception:
			# the broker forgot us (restart or eviction) — re-join instead of vanishing
			if exception.code == 404:
				try:
					__node_id__ = register_with_broker()
					logger.info('Re-joined the swarm as node {}'.format(__node_id__), __name__)
				except (urllib.error.URLError, OSError) as rejoin_exception:
					logger.warn(translate_error(rejoin_exception), __name__)
			else:
				logger.warn(translate_error(exception), __name__)
		except (urllib.error.URLError, OSError) as exception:
			logger.warn(translate_error(exception), __name__)


def leave_swarm() -> None:
	global __node_id__

	__stop_event__.set()

	if __node_id__:
		try:
			request_broker('DELETE', '/swarm/nodes/{}'.format(__node_id__))
			logger.info('Left the swarm', __name__)
		except (urllib.error.URLError, OSError):
			pass
		__node_id__ = None


def translate_error(exception : Exception) -> str:
	return 'Unable to reach the swarm at {} ({})'.format(resolve_broker_url(), exception)
