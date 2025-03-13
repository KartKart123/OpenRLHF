import time
import ray
import requests
import torch

from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)


def request_api_wrapper(url, data, score_key="rewards", try_max_times=1):
    """Synchronous request API wrapper"""
    headers = {
        "Content-Type": "application/json",
    }
    for _ in range(try_max_times):
        try:
            response = requests.post(url=url, json=data, headers=headers, timeout=1800)
            response.raise_for_status()  # Raise an HTTPError for bad responses
            response = response.json()
            assert score_key in response, f"{score_key} not in {response}"
            return response.get(score_key)
        except requests.RequestException as e:
            logger.info(f"Request error, please check: {e}")
        except Exception as e:
            logger.info(f"Unexpected error, please check: {e}")
        time.sleep(1)
    return None
    raise Exception(f"Request error for {try_max_times} times, returning None. Please check the API server.")

def request_api_wrapper_tools(url, data, score_key="rewards", tool_key = "tool_answers", try_max_times=5):
    """Synchronous request API wrapper with tool-use support"""
    headers = {
        "Content-Type": "application/json",
    }
    for _ in range(try_max_times):
        try:
            response = requests.post(url=url, json=data, headers=headers, timeout=600)
            response.raise_for_status()  # Raise an HTTPError for bad responses
            response = response.json()
            assert score_key in response, f"{score_key} not in {response}"
            assert tool_key in response, f"{tool_key} not in {response}"
            return response.get(score_key), response.get(tool_key)
        except requests.RequestException as e:
            logger.info(f"Request error, please check: {e}")
        except Exception as e:
            logger.info(f"Unexpected error, please check: {e}")
        time.sleep(1)

    raise Exception(f"Request error for {try_max_times} times, returning None. Please check the API server.")

def remote_rm_fn(api_url, queries, prompts, labels, metadata = None, score_key="rewards"):
    """remote reward model API
    api_url: RM API, We assume that the API supports two modes: merging query + response and not merging
    queries: query+response with the template
    design is made optional.
    score_key: RM score key
    """
    # scores = request_api_wrapper(api_url, {"query": queries, "prompts": prompts, "labels": labels}, score_key)
    scores = request_api_wrapper(api_url, {"query": queries, "prompts": prompts, "labels": labels, "metadata": metadata}, score_key)
    if scores is None:
        return torch.zeros(len(queries))
    return torch.tensor(scores)

def remote_rm_fn_tool(api_url, queries, prompts, labels, score_key="rewards", tool_key="tool_answers"):
    scores, tool_answers = request_api_wrapper_tools(api_url, {"query": queries, "prompts": prompts, "labels": labels}, score_key, tool_key)
    return torch.tensor(scores), tool_answers

@ray.remote
def remote_rm_fn_ray(api_url, queries, prompts, labels, metadata = None, score_key="rewards"):
    # return remote_rm_fn(api_url, queries, prompts, labels, score_key)
    return remote_rm_fn(api_url, queries, prompts, labels, metadata, score_key)

@ray.remote
def remote_rm_fn_ray_tool(api_url, queries, prompts, labels, score_key="rewards", tool_key="tool_answers"): # TODO: pass in global step
    return remote_rm_fn_tool(api_url, queries, prompts, labels, score_key, tool_key)

if __name__ == "__main__":
    # test utils
    url = "http:xxx/get_rm_score"
    score = remote_rm_fn(url, ["example query"], ["example response"])
    print(score)
