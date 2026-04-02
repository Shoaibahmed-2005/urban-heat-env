import os
import json
import time
import requests
import re
import random
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN or HF_TOKEN == "your_hf_token_here":
    print("Warning: HF_TOKEN is not correctly set. Please set it in the .env file.")

# Create OpenAI-compatible client
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "dummy_token")

ENV_URL = "http://localhost:8000"

def get_state():
    response = requests.get(f"{ENV_URL}/state")
    response.raise_for_status()
    return response.json()

def reset_env():
    response = requests.post(f"{ENV_URL}/reset")
    response.raise_for_status()
    return response.json()

def step_env(action_data):
    response = requests.post(f"{ENV_URL}/step", json=action_data)
    response.raise_for_status()
    return response.json()

def get_tasks():
    response = requests.get(f"{ENV_URL}/tasks")
    response.raise_for_status()
    return response.json()

def grade_task(task_id):
    response = requests.get(f"{ENV_URL}/grade/{task_id}")
    response.raise_for_status()
    return response.json()

def format_prompt(state, task_id):
    grid = state.get('grid', [])
    budget = state.get('budget', 0)
    
    prompt = f"Current budget: {budget}\nTask: {task_id}\nGrid state (Row, Col):\n"
    for r in range(8):
        for c in range(8):
            cell = grid[r][c]
            prompt += (f"R{r}C{c}: {cell['surface_type']}, "
                       f"Temp: {cell['temperature']}°C, "
                       f"Pop_density: {cell['population_density']}\n")
            
    return prompt

def main():
    print("Waiting for environment server to start...")
    for _ in range(10):
        try:
            requests.get(f"{ENV_URL}/health")
            break
        except requests.exceptions.ConnectionError:
            time.sleep(1)
    else:
        print("Failed to connect to the environment server.")
        return

    try:
        tasks = get_tasks()
    except Exception as e:
        print(f"Error fetching tasks: {e}")
        return

    final_scores = {}
    
    system_prompt = (
        "You are a city planning agent. Your goal is to reduce urban heat by placing "
        "cooling interventions on a city grid. Respond ONLY with valid JSON: "
        "{\"row\": <0-7>, \"col\": <0-7>, \"intervention_type\": \"green_roof\" or "
        "\"reflective_surface\" or \"tree_canopy\"}. Choose cells with high temperature "
        "and high population density. Avoid cells where budget is insufficient."
    )
    
    for task in tasks:
        task_id = task['id']
        print(f"\n=========================================")
        print(f" Starting Task: {task_id}")
        print(f"=========================================")
        reset_env()
        
        for step_idx in range(15):
            state = get_state()
            if state.get("episode_done"):
                print(f"Episode done early at step {step_idx}")
                break
                
            prompt = format_prompt(state, task_id)
            parsed_action = {}
            
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=150,
                    temperature=0.2,
                )
                
                content = response.choices[0].message.content
                # Attempt to extract JSON from the text
                match = re.search(r'\{[^{}]*\}', content)
                if match:
                    parsed_action = json.loads(match.group(0))
                else:
                    parsed_action = json.loads(content)
                    
            except Exception as e:
                print(f"[Warning] LLM error or invalid JSON: {e}. Using fallback random action.")
                # Fallback to random action if parsing fails or LLM gives a bad response
                parsed_action = {
                    "row": random.randint(0, 7), 
                    "col": random.randint(0, 7), 
                    "intervention_type": random.choice(["green_roof", "reflective_surface", "tree_canopy"])
                }
            
            # Format action for the API
            action_data = {
                "task_id": task_id,
                "row": int(parsed_action.get("row", random.randint(0, 7))),
                "col": int(parsed_action.get("col", random.randint(0, 7))),
                "intervention_type": str(parsed_action.get("intervention_type", "reflective_surface"))
            }
            
            # Ensure coordinates are bounded
            action_data["row"] = max(0, min(7, action_data["row"]))
            action_data["col"] = max(0, min(7, action_data["col"]))

            # Validate intervention_type
            if action_data["intervention_type"] not in ["green_roof", "reflective_surface", "tree_canopy"]:
                action_data["intervention_type"] = "reflective_surface"
            
            try:
                obs = step_env(action_data)
                reward = obs.get("reward", 0)
                print(f"Step {step_idx+1:2d}: Placed {action_data['intervention_type']} "
                      f"at ({action_data['row']}, {action_data['col']}) | Reward: {reward:.4f}")
                
                if obs.get("done"):
                    print("Episode finished (out of budget or max steps reached).")
                    break
            except Exception as e:
                print(f"Error during step: {e}")
                break
                
        # Get final score from the grader for this task
        try:
            result = grade_task(task_id)
            score = result.get('score', 0.0)
            final_scores[task_id] = score
            print(f">>> Final Score for {task_id}: {score:.2f} <<<")
        except Exception as e:
            print(f"Error grading task {task_id}: {e}")

    print("\n--- Summary ---")
    total = 0
    for i, (tid, score) in enumerate(final_scores.items()):
        print(f"Task {i+1} ({tid}): {score:.2f}")
        total += score
    if final_scores:
        print(f"Average: {(total/len(final_scores)):.2f}")

if __name__ == "__main__":
    main()
