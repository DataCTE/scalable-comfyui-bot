import requests
import threading
import os
import time
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import json

def perform_search(text="studio ghibli", recursion=0):
    if recursion > 10:
        return
    # force alnum
    results = requests.get(f"https://imageapi.same.energy/search?text=art masterpiece best quality Atmospheric  {text}&n=100")
    
    if results.status_code == 200:
        if len(results.text.splitlines()) > 1:
            # This result was not cached
            try:
                results = json.loads(results.text.splitlines()[2])
            except Exception as e:
                results = json.loads(results.text.splitlines()[1])
                split_1 = results["message"].split("The word '")[1]
                word = split_1.split("' isn't recognized")[0]
                return perform_search(text.replace(word, ""), recursion=recursion+1)
        else:
            # This result was cached
            results = results.json()

    downloaded_results = []
    all_threads = []

    lock = threading.Lock()

    def download_results(result):
        try:
            # https://blobcdn.same.energy/thumbnails/blobs/b/85/d1/85d18f3edbda90f75a2678cb33ef0206f39f3a57
            
            # {"id": "yPnSx", "sha1": "dc16001f7e9aa0a61c61b0ae0437f2d868c07a2a", "prefix": "a", "width": 512, "height": 723, "metadata": {"source": "pinterest", "caption": null, "title": "anime: totoro", "nsfw": false, "post_url": "https://www.pinterest.com/pin/445786063091498852/", "original_url": "http://matome.naver.jp/odai/2134250139839101201/2134252791441298003", "tags": {}}}
            # https://blobcdn.same.energy/thumbnails/blobs/a/dc/16/dc16001f7e9aa0a61c61b0ae0437f2d868c07a2a

            sha1 = result['sha1']
            prefix = result['prefix']
            
            url = f"https://blobcdn.same.energy/{prefix}/{sha1[:2]}/{sha1[2:4]}/{sha1}"

            request = requests.get(url)
            if request.status_code == 200:
                # Create an in-memory image from the downloaded bytes
                # img = Image.open(BytesIO(request.content))
                # Check if both dimensions are at least 512x512
                # if img.width >= 512 and img.height >= 512:
                with lock:
                    downloaded_results.append(request.content)
        except Exception as e:
            print(e)

    NUM_OF_RESULTS_TO_INTERROGATE = 20
    for result in results['payload']['images'][0:NUM_OF_RESULTS_TO_INTERROGATE]:
        # Create a new thread to download all results simulateniously
        thread = threading.Thread(target=download_results, args=(result,))
        thread.start()
        all_threads.append(thread)

    # Wait 15 seconds maximum 
    MAX_WAIT_TIME = 15
    start_time = time.time()
    for thread in tqdm(all_threads):
        thread.join(MAX_WAIT_TIME)
        end_time = time.time()
        
        if end_time - start_time > MAX_WAIT_TIME:
            break

    # Save all downloaded images
    os.makedirs("images", exist_ok=True)
    # Clear directory
    for f in os.listdir("images"):
        os.remove(os.path.join("images", f))

    print("Saving")
    with lock:
        for i, result in enumerate(downloaded_results):
            with open(f"images/{i}.jpg", "wb") as f:
                f.write(result)
    print(f"Done, wrote to {len(downloaded_results)} images")
    # Forcefully terminate all threads
    for thread in all_threads:
        thread.join(0)
    
    return downloaded_results
