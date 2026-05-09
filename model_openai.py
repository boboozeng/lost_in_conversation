from openai import OpenAI, AzureOpenAI
import os, time, json, re, warnings
from model_router import ModelRouter

def format_messages(messages, variables={}):
    last_user_msg = [msg for msg in messages if msg["role"] == "user"][-1]

    for k, v in variables.items():
        key_string = f"[[{k}]]"
        if key_string not in last_user_msg["content"]:
            print(f"[prompt] Key {k} not found in prompt; effectively ignored")
        assert type(v) == str, f"[prompt] Variable {k} is not a string"
        last_user_msg["content"] = last_user_msg["content"].replace(key_string, v)

    # find all the keys that are still in the prompt using regex [[STR]] where STR is alnum witout space
    keys_still_in_prompt = re.findall(r"\[\[([^\]]+)\]\]", last_user_msg["content"])
    if len(keys_still_in_prompt) > 0:
        print(f"[prompt] The following keys were not replaced: {keys_still_in_prompt}")

    return messages

class OpenAI_Model:
    def __init__(self):
        self.router = ModelRouter()

    def cost_calculator(self, model, usage, is_batch_model=False):
        is_finetuned, base_model = False, model
        if model.startswith("ft:gpt"):
            is_finetuned = True
            base_model = model.split(":")[1]

        prompt_tokens = usage['prompt_tokens']
        if 'prompt_tokens_details' in usage:
            prompt_tokens_cached = usage['prompt_tokens_details']['cached_tokens']
        else:
            prompt_tokens_cached = 0
        prompt_tokens_non_cached = prompt_tokens - prompt_tokens_cached

        completion_tokens = usage['completion_tokens']
        if base_model.startswith("gpt-4o-mini"):
            if is_finetuned:
                inp_token_cost, out_token_cost = 0.0003, 0.00015
            else:
                inp_token_cost, out_token_cost = 0.00015, 0.0006
        elif base_model.startswith("gpt-4o"):
            if is_finetuned:
                inp_token_cost, out_token_cost = 0.00375, 0.015
            else:
                inp_token_cost, out_token_cost = 0.0025, 0.01
        elif base_model.startswith("gpt-3.5-turbo"):
            inp_token_cost, out_token_cost = 0.0005, 0.0015
        elif base_model.startswith("o1-mini"):
            inp_token_cost, out_token_cost = 0.003, 0.012
        elif base_model.startswith("gpt-4.5-preview"):
            inp_token_cost, out_token_cost = 0.075, 0.150
        elif base_model.startswith("o1-preview") or base_model == "o1":
            inp_token_cost, out_token_cost = 0.015, 0.06
        ### Add new model pricing
        ### GPT Series
        elif base_model.startswith("gpt-5.5"):
            inp_token_cost, out_token_cost = 0.0015, 0.012
        elif base_model.startswith("gpt-5.4-mini"):
            inp_token_cost, out_token_cost = 0.000225, 0.00135
        elif base_model.startswith("gpt-5.4"):
            inp_token_cost, out_token_cost = 0.00075, 0.0045
        elif base_model.startswith("gpt-5.2"):
            inp_token_cost, out_token_cost = 0.000525, 0.00420
        ### GLM Series
        elif base_model.startswith("glm-5.1"):
            inp_token_cost, out_token_cost = 0.006, 0.024
        ### DeepSeek Series
        elif base_model.startswith("deepseek"):
            inp_token_cost, out_token_cost = 0.003, 0.006
        ### LLaMA Series
        elif base_model.startswith("llama"):
            inp_token_cost, out_token_cost = 0, 0
        ### Qwen Series
        elif base_model.startswith("qwen"):
            inp_token_cost, out_token_cost = 0, 0
        ### Unknown models - use gpt-4o pricing as default, but print a warning
        else:
            warnings.warn(f"[cost] Model {model} pricing unknown, using default (gpt-4o pricing)", RuntimeWarning)
            inp_token_cost, out_token_cost = 0.0025, 0.01

        cache_discount = 0.5 # cached tokens are half the price
        batch_discount = 0.5 # batch API is half the price
        total_usd = ((prompt_tokens_non_cached + prompt_tokens_cached * cache_discount) / 1000) * inp_token_cost + (completion_tokens / 1000) * out_token_cost
        if is_batch_model:
            total_usd *= batch_discount

        return total_usd

    def generate(self, messages, model="gpt-4o-mini", timeout=30, max_retries=3, temperature=1.0, is_json=False, return_metadata=False, max_tokens=None, variables={}, extra_body=None):
        kwargs = {}
        if is_json:
            kwargs["response_format"] = { "type": "json_object" }
        if extra_body is not None:
            kwargs["extra_body"] = extra_body
        N = 0

        messages = format_messages(messages, variables)

        ### Get the appropriate client for the model using the router
        requested_model = model
        routed_model, client = self.router.get_client(requested_model)

        # o1- models do not support system message. If the first message is a system message, and the second message is a user message, then prepend the user message with the system message.
        if routed_model.startswith("o1") and len(messages) > 1 and messages[0]["role"] == "system" and messages[1]["role"] == "user":
            system_message = messages[0]["content"]
            messages[1]["content"] = f"System Message: {system_message}\n{messages[1]['content']}"
            messages = messages[1:]

        while True:
            try:
                response = client.chat.completions.create(model=routed_model, messages=messages, timeout=timeout, max_completion_tokens=max_tokens, temperature=temperature, **kwargs)
                break
            except Exception as e:
                N += 1
                print(f"[retry] requested_model={requested_model}, routed_model={routed_model}, attempt={N}/{max_retries}, error={type(e).__name__}: {e}")
                if N >= max_retries:
                    raise RuntimeError(f"Failed to get response via routed model {routed_model} after {max_retries} attempts") from e
                else:
                    time.sleep(min(2 ** N, 30))

        response = response.to_dict()
        usage = response['usage']
        response_text = response["choices"][0]["message"]["content"]
        total_usd = self.cost_calculator(routed_model, usage)
        prompt_tokens_cached = 0
        if 'prompt_tokens_details' in usage:
            prompt_tokens_cached = usage['prompt_tokens_details']['cached_tokens']

        if not return_metadata:
            return response_text
        return {"message": response_text, "total_tokens": usage['total_tokens'], "prompt_tokens": usage['prompt_tokens'], "prompt_tokens_cached": prompt_tokens_cached, "completion_tokens": usage['completion_tokens'], "total_usd": total_usd}

    def generate_json(self, messages, model="gpt-4o-mini", **kwargs):
        response = self.generate(messages, model, is_json=True, **kwargs)
        response["message"] = json.loads(response["message"])
        return response


model = OpenAI_Model()
generate = model.generate
generate_json = model.generate_json





if __name__ == "__main__":
    messages = [
        {"role": "user", "content": "Humor is a way to make people laugh. Tell me a joke about UC Berkeley."},
        {"role": "assistant", "content": '{"joke": '}
    ]

    model = "gpt-4o"
    response = generate(messages, model=model, return_metadata=True)
    
    print(response)

    # batch_id = client.generate_w_delay(messages, model="gpt-4o-mini")
    # print(batch_id)
