from clip_retrieval.clip_client import ClipClient, Modality
import requests
import threading
import os
import time
from PIL import Image
from io import BytesIO
from tqdm import tqdm


client = ClipClient(url="http://localhost:1234//knn-service", indice_name="fondant_datacomp_small", aesthetic_score=9)

def perform_search(text="studio ghibli"):
    results = client.query(text=text)

    downloaded_results = []
    all_threads = []

    lock = threading.Lock()

    def download_results(result):
        try:
            request = requests.get(result["url"])
            if request.status_code == 200:
                # Create an in-memory image from the downloaded bytes
                img = Image.open(BytesIO(request.content))
                # Check if both dimensions are at least 512x512
                if img.width >= 512 and img.height >= 512:
                    with lock:
                        downloaded_results.append(request.content)
        except Exception as e:
            print(e)

    NUM_OF_RESULTS_TO_INTERROGATE = 50
    for result in results[0:NUM_OF_RESULTS_TO_INTERROGATE]:
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
