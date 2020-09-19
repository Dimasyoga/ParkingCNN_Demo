import aiohttp                          # For asynchronously making HTTP requests
import asyncio
import concurrent.futures               # Allows creating new processes
from multiprocessing import cpu_count   # Returns our number of CPU cores
                                        # Helps divide up our requests evenly across our CPU cores
from math import floor
import numpy as np
import os
import cv2
import requests
import timeit
import tflite_runtime.interpreter as tflite
from Preprocessing.Preprocessing import Preprocessing
import json
import sys

model_path = "./model.tflite"

# Load TFLite model and allocate tensors.
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']

# with open('maskParam_Labtek8Timur.json', 'r') as f:
with open('maskParam.json', 'r') as f:
    Param = json.load(f)

maskParam = []

for m in Param:
    x = []
    for p in m:
        x.append(tuple(p))
    maskParam.append(x)

pre = Preprocessing()
# pre.setImage(frame)
# pre.addMask()
pre.setMask(maskParam)

cam_addr_list = ["http://192.168.88.133", "http://192.168.88.246",
                "http://192.168.88.136", "http://192.168.88.228",
                "http://192.168.88.123", "http://192.168.88.124",
                "http://192.168.88.143", "http://192.168.88.254",
                "http://192.168.88.13", "http://192.168.88.214",
                "http://192.168.88.33", "http://192.168.88.24",
                "http://192.168.88.3", "http://192.168.88.22",
                "http://192.168.88.23", "http://192.168.88.2",
                "http://192.168.88.133", "http://192.168.88.246",
                "http://192.168.88.136", "http://192.168.88.228",
                "http://192.168.88.123", "http://192.168.88.124",
                "http://192.168.88.143", "http://192.168.88.254",
                "http://192.168.88.13", "http://192.168.88.214",
                "http://192.168.88.33", "http://192.168.88.24",
                "http://192.168.88.3", "http://192.168.88.22",
                "http://192.168.88.23", "http://192.168.88.2",
                "http://192.168.88.43", "http://192.168.88.16",
                "http://192.168.88.11", "http://192.168.88.26",
                "http://192.168.88.33", "http://192.168.88.24",
                "http://192.168.88.3", "http://192.168.88.22",
                "http://192.168.88.23", "http://192.168.88.2",
                "http://192.168.88.43", "http://192.168.88.16",
                "http://192.168.88.11", "http://192.168.88.26",
                "http://192.168.88.33", "http://192.168.88.24",
                "http://192.168.88.23", "http://192.168.88.2",
                "http://192.168.88.136", "http://192.168.88.228",
                "http://192.168.88.2",]

# cam_addr_list = ["http://192.168.88.133", "http://192.168.88.246"]

img_array = np.empty(len(cam_addr_list), dtype=object)

timeout = aiohttp.ClientTimeout(total=1.5)

async def get_response(url):
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.get(url+"/capture") as response:
                # response.raise_for_status()
                print(f"Response status ({url}): {response.status}")
                image = np.asarray(bytearray(await response.read()), dtype="uint8")
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")
            image = np.zeros((1600, 1200, 3), dtype="uint8")
        
        except aiohttp.ClientConnectorError as e:
            print('Connection Error', str(e))
            image = np.zeros((1600, 1200, 3), dtype="uint8")
            
        except Exception as err:
            print(f"An error ocurred: {err}")
            # print(str((cam_addr_list[index]+"/capture")))
            image = np.zeros((1600, 1200, 3), dtype="uint8")
        
        # img_array[index] = image

        pre.setImage(image)
        crop = pre.getCrop()

        for frame in crop:
            frame = cv2.resize(frame, (input_shape[1], input_shape[2]), interpolation = cv2.INTER_CUBIC)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32)
            frame = frame / 255.
            frame = np.expand_dims(frame, 0)

            interpreter.set_tensor(input_details[0]['index'], frame)
            interpreter.invoke()
        
        return url, interpreter.get_tensor(output_details[0]['index'])
        


async def request(indexs):
    result = await asyncio.gather(*[get_response(cam_addr_list[i]) for i in indexs])
    return result

#     if index == 0:
#         task = []
#         for i in range(5):
#             task.append(get_response(i))

#         result = await asyncio.gather(
#             *task
#         )
#     elif index == 1:
#         result = await asyncio.gather(
#             get_response(5),
#             get_response(6),
#             get_response(7),
#             get_response(8),
#             get_response(9),
#         )
#     elif index == 2:
#         result = await asyncio.gather(
#             get_response(10),
#             get_response(11),
#             get_response(12),
#             get_response(13),
#             get_response(14),
#         )
#     elif index == 3:
#         result = await asyncio.gather(
#             get_response(15),
#             get_response(16),
#             get_response(17),
#             get_response(18),
#             get_response(19),
#         )

#     return result

def start_request(indexs):
    # return asyncio.run(request(index))
    # res = []
    # for i in range (3):
    #     loop = asyncio.get_event_loop()
    #     result = {"iter":i, "result":loop.run_until_complete(request(indexs))}
    #     res.append(result)
    # return res
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(request(indexs))

# def main():
#     for i in range(cpu_count()):
#         start_request(i)

def main():
    # NUM_CORES = cpu_count()-1 # Our number of CPU cores (including logical cores)
    # NUM_URL = len(cam_addr_list)
    # print("url count: {0} cpu count: {1}".format(NUM_URL, NUM_CORES))
    # URL_PER_CORE = floor(NUM_URL / NUM_CORES)

    # futures = [] # To store our futures

    # with concurrent.futures.ProcessPoolExecutor(cpu_count()) as executor:
    #     for i in range(NUM_CORES):
    #         indexs = []
    #         for j in range(URL_PER_CORE*i, URL_PER_CORE*(i+1)):
    #             print("j{0} : {1}".format(i, j))
    #             indexs.append(j)

    #         new_future = executor.submit(
    #             start_request, # Function to perform
    #             # v Arguments v
    #             indexs
    #         )
    #         futures.append(new_future)
        
    #     indexs = []
    #     for j in range(URL_PER_CORE*NUM_CORES, NUM_URL):
    #         print("j{0} : {1}".format(NUM_CORES, j))
    #         indexs.append(j)

    #     if (NUM_URL%NUM_CORES != 0):
    #         futures.append(
    #             executor.submit(
    #                 start_request,
    #                 indexs
    #             )
    #         )

    # concurrent.futures.wait(futures)
    # print(futures[0].result())
    # print(len(futures))

    NUM_CORES = cpu_count() # Our number of CPU cores (including logical cores)
    NUM_URL = len(cam_addr_list)
    URL_PER_CORE = floor(NUM_URL / NUM_CORES)
    REMAINDER = NUM_URL % NUM_CORES

    print("url count: {0} cpu count: {1} count {2} remainder {3}".format(NUM_URL, NUM_CORES, URL_PER_CORE, REMAINDER))

    futures = [] # To store our futures

    with concurrent.futures.ProcessPoolExecutor(NUM_CORES) as executor:
        for i in range(NUM_CORES):
            indexs = []
            if (i < REMAINDER):
                start = i * (URL_PER_CORE + 1)
                stop = start + (URL_PER_CORE + 1)
                
            else:
                start = (i * URL_PER_CORE) + REMAINDER
                stop = start + URL_PER_CORE


            for j in range(start, stop):
                # print("j{0} : {1}".format(i, j))
                indexs.append(j)

            print("core {0} get {1} task from {2} to {3}".format(i, len(indexs), start, stop))
            
            new_future = executor.submit(
                start_request, # Function to perform
                # v Arguments v
                indexs
            )
            futures.append(new_future)

    concurrent.futures.wait(futures)
    print(futures[0].result())
    print(len(futures))

if __name__ == "__main__":
    print(timeit.timeit(main, number=10))
    # main()
    # print(img_array)
    # for i,img in enumerate(img_array):
    #     cv2.imshow(str(i), img)
    
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


