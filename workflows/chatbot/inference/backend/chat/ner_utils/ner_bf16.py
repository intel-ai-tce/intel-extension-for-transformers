import datetime
from datetime import timezone, timedelta
from threading import Thread
import os
import re
import torch
import spacy
import time
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    AutoConfig,
)
import intel_extension_for_pytorch as intel_ipex

def set_cpu_running_env():
    os.environ["ONEDNN_MAX_CPU_ISA"] = "AVX512_CORE_BF16"

nlp = spacy.load("en_core_web_lg")

model_name ="/home/tme/Llama-2-7b-chat-hf/"
print(f"Starting to load the model {model_name} into memory")


# ============== set up env and prepare models and configs ================
set_cpu_running_env()
torch_dtype = torch.bfloat16
config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
config.init_device = 'cuda:0' if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(
    model_name,
    use_fast=False if (re.search("llama", model_name, re.IGNORECASE)
        or re.search("neural-chat-7b-v2", model_name, re.IGNORECASE)) else True
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    config=config,
    device_map="auto",
)
model = intel_ipex.optimize(
    model.eval(),
    dtype=torch_dtype,
    inplace=True,
    level="O1",
    auto_kernel_selection=True,
)
model.generation_config.pad_token_id = 0
model.generation_config.bos_token_id = 1
model.generation_config.eos_token_id = 2
tok.pad_token_id = model.generation_config.pad_token_id
tok.eos_token_id = model.generation_config.eos_token_id
tok.bos_token_id = model.generation_config.bos_token_id
print(f"Successfully loaded the bf16 model {model_name} into memory")

month_date_list = [31, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30]


# ================= util functions ===================
def check_query_time(query, cur_time):
    # prompt = """Please determine the precise time mentioned in the user's query. Your response should consist only of an accurate time in the format 'Time: YYYY-MM-DD' or 'Period: YYYY-MM-DD to YYYY-MM-DD.' If the user query does not include any time reference, please reply with 'None'.
    # \n\n###Current Time:\n{}\n\nUser Query:\n{}\n\nResponse:\n""".format(cur_time, query)
    prompt = """### Instruction: Please thoughtfully identify the precise time range mentioned in the user's query based on the given current time. The response should follows the following requirements. \n
    ### Requirements:
    1. Your response should consist only of an accurate time in the format 'Time: YYYY-MM-DD' or 'Period: YYYY-MM-DD to YYYY-MM-DD.' 
    2. Please carefully check the accuracy of the identifiction results. 
    3. The phrase "in the last month" means "in the thirty or so days up to and including today".\n
    ### Current Time:\n{}\n
    ### User Query:\n{}\n
    ### Response:\n""".format(cur_time, query)

    return prompt


def enforce_stop_tokens(text: str) -> str:
    """Cut off the text as soon as any stop words occur."""
    stopwords = ["</s"]
    return re.split("|".join(stopwords), text)[0]


# ================= inference =================
def inference(query):
    SHA_TZ = timezone(
        timedelta(hours=8),
        name='Asia/Shanghai'
    )
    utc_now = datetime.datetime.utcnow().replace(tzinfo=timezone.utc)
    cur_time = utc_now.astimezone(SHA_TZ).strftime("%Y/%m/%d")
    print("current time is:{}".format(cur_time))
    prompt = check_query_time(query, cur_time)
    inputs= tok(prompt, return_token_type_ids=False, return_tensors="pt")
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=False)
    
    cur_time = time.time()

    # inference for bf16
    generate_kwargs = dict(
        max_new_tokens=32,
        temperature=0.01,
        top_k=3,
        repetition_penalty=1.1,
    )

    def generate_output():
        dtype = model.dtype if hasattr(model, 'dtype') else torch.bfloat16
        try:
            with torch.no_grad():
                context = torch.cpu.amp.autocast(enabled=True, dtype=dtype, cache_enabled=True)
                with context:
                    output_token=model.generate(
                        **inputs,
                        **generate_kwargs,
                        streamer=streamer,
                        return_dict_in_generate=True,
                    )
                    return output_token
        except Exception as e:
            raise Exception(e)

    thread = Thread(target=generate_output)
    thread.start()
    text = ""
    for new_text in streamer:
        text += new_text
    text = enforce_stop_tokens(text)

    print(f'inference time: {time.time() - cur_time}')
    cur_time = time.time()

    doc = nlp(text)
    mentioned_time = {"time":[], "period":[]}
    for ent in doc.ents:
        if ent.label_ == 'DATE':
            if bool(re.search(r'\d', str(ent))):
                print("The target time is {}".format(ent))
                if "to" in text:
                    print(f"text: {text}")
                    print(f"ent: {ent}")
                    if "to" in ent.text:
                        cur_periods = ent.text.split(" to ")
                        mentioned_time['period'].extend(cur_periods)
                    else:
                        if len(mentioned_time["period"]) > 0 and mentioned_time["period"][-1] == ent.text:
                            mentioned_time["period"].pop()
                        else:
                            mentioned_time["period"].append(ent.text)
                else:
                    mentioned_time["time"].append(ent.text)
    print("mentioned_time: ", mentioned_time)
    print(len(mentioned_time["period"]))
    if len(mentioned_time["period"]) % 2 != 0:
        mentioned_time["time"] = list(set(mentioned_time["time"]+mentioned_time["period"]))
        mentioned_time["period"] = []

    new_doc = nlp(query)
    location = []
    name = []
    organization = []
    s_time = []
    for ent in new_doc.ents:
        if (ent.label_ == 'GPE'):
            location.append(ent.text)
        elif (ent.label_ == 'LOC'):
            location.append(ent.text)
        elif (ent.label_ == 'PERSON'):
            name.append(ent)
        elif (ent.label_ == 'ORG'):
            organization.append(ent)
        elif (ent.label_ == 'DATE' or ent.label_ == 'TIME'):
            s_time.append(ent)
    if s_time == []:
        mentioned_time = {"time": [], "period": []}
    location = list(set(location))

    result_period = []
    for sub in range(len(mentioned_time['period'])//2):
        from_time = mentioned_time['period'][2*sub]
        to_time = mentioned_time['period'][2*sub+1]
        result_period.append({"from": from_time, "to": to_time})
    if 'last month' in query:
        to_time = datetime.datetime.today()
        now_month = to_time.month
        from_time = to_time - timedelta(days=month_date_list[now_month-1])
        result_period = [{"from": str(from_time)[:10], "to": str(to_time)[:10]}]
    result = {"period": result_period, "time": mentioned_time['time'], 'location': location, "name": name, "organization": organization}

    return result


# CUDA_VISIBLE_DEVICES=0 python test.py
if __name__ == "__main__":

    while True:
        query = input("Enter query (or 'exit' to quit): ")
        if query == 'exit':
            print('exit')
            break
        start_time = time.time()
        result = inference(query)
        print(result)
        # time_period, time_point, locations, names, organizations = result['period'], result['time'], result['location'], result['name'], result['organization']
        # if time_period:
        #     for sub in range(len(time_period)//2):
        #         print("The target time period of query: {} is from {} to {}.".format(query, time_period[2*sub], time_period[2*sub+1]))
        # elif time_point:
        #     for sub in range(len(time_point)):
        #         print("The target time point of query: {} is {}.".format(query, time_point[sub]))
        # if locations:
        #     for loc in locations:
        #         print("The target location of query: {} is {}.".format(query, loc))
        # if names:
        #     for na in names:
        #         print("The mentioned name in the query: {} is {}.".format(query, na))
        # if organizations:
        #     for org in organizations:
        #         print("The mentioned organization in the query: {} is {}.".format(query, org))

        end_time = time.time()
        print("Inference cost {} seconds.".format(end_time - start_time))