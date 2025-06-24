import httpx

endpoint = "/generate"
base_url = "https://1fc4-34-124-176-101.ngrok-free.app"
url = base_url + endpoint
headers= {"Content-Type": "application/json"}

def stream_story(input_text:str, url:str, headers):
  payload = {"input_text": input_text}

  with httpx.stream("POST", url=url, headers=headers, json=payload, timeout=None) as response:
    if response.status_code != 200:
      print(f"Request failed with status code: {response.status_code}")
      return
    
    print("\n Generating story:\n")
    ##------------ SPACING ISSUES -------------##
    # for line in response.iter_lines():
    #   if line.startswith("data"):
    #     token = line.replace("data:", "").strip()
    #     print(line, end="", flush=True)

    ##------------ NO SPACING ISSUES -------------##
    for token in response.iter_text():
        print(token, end="", flush=True)

if __name__ == "__main__":
    prompt = "Once upon a time"
    stream_story(input_text=prompt, url=url, headers=headers)
  