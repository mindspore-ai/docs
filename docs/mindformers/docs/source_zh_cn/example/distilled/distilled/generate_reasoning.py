"""Generate reasoning completions using a language model API."""
import traceback
import argparse
import asyncio
import hashlib
import json
import os
import random
from typing import Set

from tqdm.asyncio import tqdm

import aiofiles
import aiohttp
import uvloop
from datasets import load_dataset

file_lock = asyncio.Lock()

API_KEY = "your_api_key_here"  # 替换为你的API密钥

async def generate_completion(session, prompt, args):
    """Generate a completion using the API."""
    retry_budget = 10
    while retry_budget > 0:
        try:
            await asyncio.sleep(random.uniform(0.0, 0.1))
            async with session.post(
                    f"https://{args.api_addr}/v1/chat/completions",
                    json={
                        "model": args.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": args.max_tokens,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                    },
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {API_KEY}",
                    },
                ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    print(f"HTTP error {response.status}: {error_text}")
                    raise Exception(f"HTTP error {response.status}: {error_text}")
                return await response.json(content_type=None)
        except (aiohttp.ClientError, asyncio.TimeoutError) as e_comm:
            traceback.print_exc()
            print(f"API communication error (will retry {retry_budget-1} more times): {e_comm}")
            retry_budget -= 1
            await asyncio.sleep(10)
        except (json.JSONDecodeError, aiohttp.ContentTypeError) as e_parse:
            traceback.print_exc()
            print(f"API error (will retry): {e_parse}")
            retry_budget -= 1
            await asyncio.sleep(10)
    return None


async def process_example(example, session, args, output_file, pbar):
    """Process a single example and write the result to the output file."""
    prompt = args.prompt_template.format(prompt=example[args.prompt_column])

    try:
        tasks = [generate_completion(session, prompt, args) for _ in range(args.num_generations)]

        completions = await asyncio.gather(*tasks)

        if any(completion is None for completion in completions):
            print(f"Error processing example")
            pbar.update(1)
            return None

        generations = []
        finish_reasons = []
        api_metadata = []

        for completion in completions:
            if completion["choices"][0]["message"].get("reasoning_content") is not None:
                completion["choices"][0]["message"]["content"] = (
                    "<think>\n"
                    + completion["choices"][0]["message"].get("reasoning_content")
                    + "</think>\n"
                    + completion["choices"][0]["message"]["content"]
                )
            generations.append(completion["choices"][0]["message"]["content"])
            finish_reasons.append(completion["choices"][0]["finish_reason"])
            api_metadata.append(completion["usage"])

        result = {
            **example,
            "generations": generations,
            "finish_reasons": finish_reasons,
            "api_metadata": api_metadata,
            "prompt": prompt,
        }

        async with file_lock:
            async with aiofiles.open(output_file, mode="a") as f:
                await f.write(json.dumps(result) + "\n")
                await f.flush()

        pbar.set_postfix(active=len(pbar.active_tasks), refresh=False)
        pbar.update(1)

        return result
    except KeyError as e_key:
        err_msg = f"Error processing example due to missing key: {e_key}."
        if not prompt:
            err_msg += " Potentially accessing initial prompt data for column " \
                f"'{args.prompt_column}' in example: {str(example)[:200]}..."
        else:
            err_msg += f" For prompt: {prompt[:100]}..."
        print(err_msg)
        traceback.print_exc()
        pbar.update(1)
        return None
    except IndexError as e_idx:
        print(f"Error processing example due to list index out of bounds: {e_idx} for prompt: {prompt[:100]}...")
        traceback.print_exc()
        pbar.update(1)
        return None
    except (TypeError, ValueError) as e_fmt:
        print("Error processing example due to type or value error (e.g., in prompt formatting or data handling): " \
            f"{e_fmt} for prompt (if available): {prompt[:100]}...")
        traceback.print_exc()
        pbar.update(1)
        return None
    except (IOError, OSError) as e_io:
        print(f"Error processing example due to I/O error: {e_io} for prompt: {prompt[:100]}...")
        traceback.print_exc()
        pbar.update(1)
        return None


async def load_processed_uuids(output_file, uuid_column):
    """Load processed UUIDs from the output file."""
    processed_uuids = set()
    if os.path.exists(output_file):
        async with aiofiles.open(output_file, mode="r") as f:
            async for line in f:
                try:
                    data = json.loads(line)
                    processed_uuids.add(hashlib.md5(str(data[uuid_column]).encode()).hexdigest())
                except json.JSONDecodeError:
                    continue
    return processed_uuids


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--prompt-column", type=str, required=True)
    parser.add_argument("--uuid-column", type=str, required=True)
    parser.add_argument("--api-addr", type=str, default="localhost:39876")
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument(
        "--prompt-template",
        type=str,
        default="You will be given a problem. " \
            "Please reason step by step, and put your final answer within \\boxed{{}}:\n{prompt}",
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--max-concurrent", type=int, default=1000)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset_name, split="train").shuffle()
    processed_uuids = await load_processed_uuids(args.output_file, args.uuid_column)
    if processed_uuids:
        print(f"Found {len(processed_uuids)} already processed examples, resuming from there...")

    if not os.path.exists(args.output_file):
        async with aiofiles.open(args.output_file, mode="w") as f:
            await f.write("")

    active_tasks: Set[asyncio.Task] = set()

    pbar = tqdm(
        total=len(dataset) - len(processed_uuids),
        desc="Generating responses",
        unit="row",
        mininterval=2,
        smoothing=0.0001,
    )

    pbar.active_tasks = active_tasks
    count = 0
    async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60 * 60),
            connector=aiohttp.TCPConnector(limit=args.max_concurrent, ttl_dns_cache=300, keepalive_timeout=60 * 60),
        ) as session:
        for example in dataset:
            uuid = hashlib.md5(str(example[args.uuid_column]).encode()).hexdigest()
            if uuid not in processed_uuids:

                while len(active_tasks) >= args.max_concurrent:
                    done, active_tasks = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)
                    for task in done:
                        try:
                            await task
                        except asyncio.CancelledError:
                            print(f"A task was cancelled: {task!r}")

                task = asyncio.create_task(process_example(example, session, args, args.output_file, pbar))
                active_tasks.add(task)
                task.add_done_callback(active_tasks.discard)

                pbar.set_postfix(active=len(active_tasks), refresh=True)
            count += 1
            if count > 2:
                break

        if active_tasks:
            await asyncio.gather(*active_tasks, return_exceptions=True)

    pbar.close()


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(main())
